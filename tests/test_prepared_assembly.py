from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import pytest
from pydub import AudioSegment

from tts5703.assemble import assemble_prepared_dialogue
from tts5703.controlled_tts import map_turn_to_higgs
from tts5703.render_plan import PreparedDialogue, PreparedTurn, TurnRenderResult


def _write_speech(path: Path, duration_ms: int = 100) -> None:
    sample_rate = 8_000
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\xe8\x03" * (sample_rate * duration_ms // 1_000))


def _turn(
    tmp_path: Path,
    ordinal: int,
    source_turn_id: str,
    pause_before: str,
) -> PreparedTurn:
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference")
    source = {
        "turn_id": source_turn_id,
        "speaker": "User",
        "text": "Prepared speech.",
        "acoustic": {
            "required": {
                "rate": "normal",
                "arousal": 2,
                "affect": "neutral",
                "pause_before": pause_before,
                "pause_within": 0,
                "hesitations": 0,
            },
            "best_effort": {},
        },
    }
    context = {
        "dialogue_id": "assembly-dialogue",
        "scenario": {
            "speakers": {
                "caller": {"speaker_id": "C123"},
                "counsellor": {"speaker_id": "L456"},
            }
        },
    }
    return PreparedTurn(
        ordinal=ordinal,
        source_turn_id=source_turn_id,
        upstream_role="User",
        logical_role="caller",
        upstream_scenario_speaker_id="C123",
        render_speaker_id="spk_001",
        reference_wav="reference.wav",
        resolved_reference_wav=reference,
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        plan=map_turn_to_higgs(source, dialogue_context=context),
    )


def _dialogue(*turns: PreparedTurn) -> PreparedDialogue:
    return PreparedDialogue(
        dialogue_id="assembly-dialogue",
        record_sha256="1" * 64,
        turns=tuple(turns),
        assignment_sha256="2" * 64,
        registry_sha256="3" * 64,
        active_speakers_sha256="4" * 64,
    )


def _result(turn: PreparedTurn, path: Path) -> TurnRenderResult:
    return TurnRenderResult(
        ordinal=turn.ordinal,
        source_turn_id=turn.source_turn_id,
        output_path=path,
        synthesis_status="synthesized",
        rate_status="not_required",
    )


@pytest.mark.parametrize(
    ("pause_before", "milliseconds"),
    [("none", 0), ("short", 500), ("long", 900)],
)
def test_prepared_pause_before_is_once_and_has_no_pause_after(
    tmp_path: Path, pause_before: str, milliseconds: int
) -> None:
    turn = _turn(tmp_path, 1, "non-numeric-id", pause_before)
    path = tmp_path / "speech.wav"
    _write_speech(path)
    original_bytes = path.read_bytes()

    audio, timings = assemble_prepared_dialogue(
        _dialogue(turn),
        [_result(turn, path)],
        {"fade_ms": 5, "pause": {"default_ms": 7_777}},
    )

    assert len(audio) == milliseconds + 100
    assert timings == [
        {
            "ordinal": 1,
            "source_turn_id": "non-numeric-id",
            "pause_before_ms": milliseconds,
            "pause_after_ms": 0,
            "start_sec": milliseconds / 1000,
            "end_sec": (milliseconds + 100) / 1000,
        }
    ]
    assert path.read_bytes() == original_bytes


def test_prepared_timestamps_and_lookup_follow_ordinal_not_source_id(
    tmp_path: Path,
) -> None:
    first = _turn(tmp_path, 1, "zeta-source", "short")
    second = _turn(tmp_path, 2, "alpha-source", "long")
    first_path = tmp_path / "arbitrary-first.wav"
    second_path = tmp_path / "arbitrary-second.wav"
    _write_speech(first_path)
    _write_speech(second_path)

    audio, timings = assemble_prepared_dialogue(
        _dialogue(second, first),
        [_result(second, second_path), _result(first, first_path)],
        {"fade_ms": 5, "pause": {"default_ms": 8_888}},
    )

    assert [
        (item["source_turn_id"], item["start_sec"], item["end_sec"]) for item in timings
    ] == [
        ("zeta-source", 0.5, 0.6),
        ("alpha-source", 1.5, 1.6),
    ]
    assert len(audio) == 1_600
    assert audio[:500].rms == 0
    assert audio[600:1500].rms == 0
    assert audio[500:600].rms > 0
    assert audio[1500:1600].rms > 0


def test_final_pause_segments_are_not_faded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    turn = _turn(tmp_path, 1, "source", "short")
    path = tmp_path / "speech.wav"
    _write_speech(path)
    fade_calls: list[tuple[str, int]] = []
    original_fade_in = AudioSegment.fade_in
    original_fade_out = AudioSegment.fade_out

    def track_fade_in(segment: AudioSegment, duration: int) -> AudioSegment:
        fade_calls.append(("in", len(segment)))
        return original_fade_in(segment, duration)

    def track_fade_out(segment: AudioSegment, duration: int) -> AudioSegment:
        fade_calls.append(("out", len(segment)))
        return original_fade_out(segment, duration)

    monkeypatch.setattr(AudioSegment, "fade_in", track_fade_in)
    monkeypatch.setattr(AudioSegment, "fade_out", track_fade_out)

    audio, _ = assemble_prepared_dialogue(
        _dialogue(turn), [_result(turn, path)], {"fade_ms": 5}
    )

    assert fade_calls == [("in", 100), ("out", 100)]
    assert audio[:500].rms == 0
    assert audio.frame_rate == 8_000
    assert audio.sample_width == 2
    assert audio.channels == 1
