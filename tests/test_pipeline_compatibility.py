"""CPU-only compatibility checks for rate mapping, assembly, and metadata."""

import wave
from itertools import pairwise
from pathlib import Path

import pytest

from tts5703.assemble import TurnTiming, assemble_dialogue
from tts5703.metadata import build_metadata
from tts5703.tts_engine import rate_to_edge_tts, rate_to_kokoro_speed
from tts5703.validate import NormalizedTurn


@pytest.mark.parametrize(
    ("semantic_rate", "edge_rate"),
    [("slow", "-20%"), ("normal", "+0%"), ("fast", "+20%")],
)
def test_semantic_rate_maps_to_edge_tts(semantic_rate: str, edge_rate: str) -> None:
    assert rate_to_edge_tts(semantic_rate) == edge_rate


@pytest.mark.parametrize(
    ("semantic_rate", "kokoro_speed"),
    [("slow", 0.8), ("normal", 1.0), ("fast", 1.2)],
)
def test_semantic_rate_maps_to_kokoro(semantic_rate: str, kokoro_speed: float) -> None:
    assert rate_to_kokoro_speed(semantic_rate) == kokoro_speed


def test_legacy_percentage_rate_is_preserved_for_edge_tts() -> None:
    assert rate_to_edge_tts("-10%") == "-10%"


def test_legacy_percentage_rate_maps_to_kokoro_speed() -> None:
    assert rate_to_kokoro_speed("+20%") == pytest.approx(1.2)


def _write_silent_wav(path: Path, duration_ms: int) -> None:
    sample_rate = 8_000
    frame_count = sample_rate * duration_ms // 1_000
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\x00\x00" * frame_count)


def _turn(
    turn_id: int,
    *,
    pause_before_ms: int = 0,
    pause_after_ms: int = 0,
) -> NormalizedTurn:
    return NormalizedTurn(
        turn_id=turn_id,
        speaker="caller",
        text=f"Turn {turn_id}",
        label="normal",
        rate="normal",
        pause_before_ms=pause_before_ms,
        pause_after_ms=pause_after_ms,
        arousal="medium",
        coarse_affect="concerned",
        emotion="worried",
        paralinguistic_events=["sigh"],
    )


def test_first_turn_pause_before_moves_speech_timestamp(tmp_path: Path) -> None:
    turn = _turn(1, pause_before_ms=125)
    turn_path = tmp_path / "turn_001.wav"
    _write_silent_wav(turn_path, 100)

    audio, timings = assemble_dialogue(
        [turn], {turn.turn_id: turn_path}, {"fade_ms": 5}
    )

    assert timings[0].start_sec == 0.125
    assert timings[0].end_sec == 0.225
    assert len(audio) == 225


def test_pauses_advance_timeline_without_overlapping_turns(tmp_path: Path) -> None:
    turns = [
        _turn(1, pause_before_ms=50, pause_after_ms=200),
        _turn(2, pause_before_ms=75, pause_after_ms=25),
        _turn(3),
    ]
    paths = {}
    for turn in turns:
        path = tmp_path / f"turn_{turn.turn_id:03d}.wav"
        _write_silent_wav(path, 100)
        paths[turn.turn_id] = path

    audio, timings = assemble_dialogue(turns, paths, {"fade_ms": 5})

    assert [(timing.start_sec, timing.end_sec) for timing in timings] == [
        (0.05, 0.15),
        (0.425, 0.525),
        (0.55, 0.65),
    ]
    assert all(
        earlier.start_sec <= earlier.end_sec <= later.start_sec
        for earlier, later in pairwise(timings)
    )
    assert len(audio) == 650


def test_metadata_preserves_requested_acoustic_spec() -> None:
    timing = TurnTiming(
        turn_id=1,
        speaker="caller",
        text="Please stay with me.",
        label="alert",
        rate="slow",
        pause_before_ms=150,
        pause_after_ms=700,
        arousal="high",
        coarse_affect="distressed",
        emotion="fearful",
        paralinguistic_events=["sigh", {"type": "sob"}],
        start_sec=0.15,
        end_sec=0.65,
    )
    turn_path = Path("turn_001.wav")

    metadata = build_metadata(
        "dialogue001",
        Path("dialogue001_clean.wav"),
        Path("dialogue001_telephone.wav"),
        [timing],
        {1: turn_path},
        {"engine": "test"},
    )

    turn_metadata = metadata["turns"][0]
    assert turn_metadata["rate"] == "slow"
    assert turn_metadata["pause_before_ms"] == 150
    assert turn_metadata["pause_after_ms"] == 700
    assert turn_metadata["arousal"] == "high"
    assert turn_metadata["coarse_affect"] == "distressed"
    assert turn_metadata["emotion"] == "fearful"
    assert turn_metadata["paralinguistic_events"] == ["sigh", {"type": "sob"}]
