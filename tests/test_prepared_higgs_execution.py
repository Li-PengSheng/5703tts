from __future__ import annotations

import asyncio
import hashlib
import logging
import subprocess
import wave
from pathlib import Path
from typing import Any

import pytest

from tts5703 import tts_engine
from tts5703.backends import higgs
from tts5703.controlled_tts import map_turn_to_higgs
from tts5703.render_models import HiggsPreparedTurn, PreparedDialogue


def _write_wav(path: Path, duration_ms: int = 200) -> None:
    sample_rate = 8_000
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\x10\x00" * (sample_rate * duration_ms // 1_000))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepared_turn(
    tmp_path: Path,
    *,
    ordinal: int = 1,
    source_turn_id: str = "source-alpha",
    rate: str = "normal",
    pause_before: str = "none",
    pause_within: int = 2,
    hesitations: int = 2,
    arousal: int = 1,
    affect: str = "warm",
) -> HiggsPreparedTurn:
    reference = tmp_path / "approved" / f"reference-{ordinal}.wav"
    _write_wav(reference, 50)
    turn = {
        "turn_id": source_turn_id,
        "speaker": "User",
        "text": "First sentence. Second sentence.",
        "acoustic": {
            "required": {
                "rate": rate,
                "arousal": arousal,
                "affect": affect,
                "pause_before": pause_before,
                "pause_within": pause_within,
                "hesitations": hesitations,
            },
            "best_effort": {},
        },
    }
    context = {
        "dialogue_id": "prepared-dialogue",
        "scenario": {
            "speakers": {
                "caller": {"speaker_id": "C123"},
                "counsellor": {"speaker_id": "L456"},
            }
        },
    }
    plan = map_turn_to_higgs(turn, dialogue_context=context)
    return HiggsPreparedTurn(
        ordinal=ordinal,
        source_turn_id=source_turn_id,
        upstream_role="User",
        logical_role="caller",
        upstream_scenario_speaker_id="C123",
        render_speaker_id="spk_001",
        reference_wav=f"approved/reference-{ordinal}.wav",
        resolved_reference_wav=reference.resolve(),
        reference_sha256=_sha(reference),
        plan=plan,
    )


def _dialogue(*turns: HiggsPreparedTurn) -> PreparedDialogue:
    return PreparedDialogue(
        dialogue_id="prepared-dialogue",
        record_sha256="1" * 64,
        turns=tuple(turns),
        assignment_sha256="2" * 64,
        registry_sha256="3" * 64,
        active_speakers_sha256="4" * 64,
    )


def _config(tmp_path: Path, *, engine: str = "higgs") -> dict[str, Any]:
    server = tmp_path / "sgl-omni"
    server.write_text("fake", encoding="utf-8")
    server.chmod(0o755)
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    return {
        "tts": {
            "engine": engine,
            "higgs": {
                "server_executable": str(server),
                "model_dir": str(model),
                "ffmpeg_bin": "/fake/ffmpeg",
            },
        }
    }


def _successful_request(captured: list[dict[str, Any]]):
    def request(worker: object, payload: dict[str, Any]) -> dict[str, Any]:
        captured.append(payload)
        output = Path(payload["output_path"])
        _write_wav(output)
        return {"status": "ok", "output_path": str(output.resolve())}

    return request


def test_cached_plan_reaches_one_exact_worker_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    turn = _prepared_turn(tmp_path)
    dialogue = _dialogue(turn)
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(higgs, "_get_worker", lambda *args: object())
    monkeypatch.setattr(higgs, "_request", _successful_request(captured))
    monkeypatch.setattr(
        higgs.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("normal rate must not invoke FFmpeg"),
    )

    tts_engine.preflight_prepared_dialogue(dialogue, _config(tmp_path))
    results = asyncio.run(
        tts_engine.synthesize_prepared_turns(dialogue, tmp_path, _config(tmp_path))
    )

    assert len(captured) == 1
    assert set(captured[0]) == {"model_input", "reference_wav", "output_path"}
    assert captured[0]["model_input"] == turn.plan["higgs"]["model_input"]
    assert captured[0]["model_input"].startswith("<|emotion:contentment|>")
    assert "ummm..." in captured[0]["model_input"]
    assert captured[0]["model_input"].count("<|prosody:pause|>") == 2
    assert captured[0]["reference_wav"] == str(turn.resolved_reference_wav)
    assert "C123" not in captured[0]["reference_wav"]
    assert "wrong" not in captured[0]["reference_wav"]
    assert results[0].source_turn_id == "source-alpha"
    assert results[0].synthesis_status == "synthesized"
    assert results[0].rate_status == "not_required"


def test_all_turns_preflight_before_first_worker_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _prepared_turn(tmp_path, ordinal=1, source_turn_id="first")
    second = _prepared_turn(tmp_path, ordinal=2, source_turn_id="second")
    second.resolved_reference_wav.unlink()
    worker_calls: list[object] = []
    monkeypatch.setattr(higgs, "_get_worker", lambda *args: worker_calls.append(args))
    monkeypatch.setattr(higgs, "_request", lambda *args: worker_calls.append(args))

    with pytest.raises(RuntimeError, match="reference audio is missing"):
        tts_engine.preflight_prepared_dialogue(
            _dialogue(first, second), _config(tmp_path)
        )

    assert worker_calls == []
    assert not list(tmp_path.glob("turn_*.wav"))


@pytest.mark.parametrize("engine", ["cosyvoice"])
def test_non_higgs_final_execution_fails_closed_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    turn = _prepared_turn(tmp_path)
    calls: list[object] = []
    monkeypatch.setattr(higgs, "_get_worker", lambda *args: calls.append(args))

    with pytest.raises(RuntimeError, match="not implemented|does not match"):
        tts_engine.preflight_prepared_dialogue(
            _dialogue(turn), _config(tmp_path, engine=engine)
        )

    assert calls == []
    assert not list(tmp_path.glob("turn_*.wav"))


def test_invalid_ordinals_fail_before_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    turn = _prepared_turn(tmp_path, ordinal=2)
    calls: list[object] = []
    monkeypatch.setattr(higgs, "_get_worker", lambda *args: calls.append(args))

    with pytest.raises(RuntimeError, match="ordinals"):
        tts_engine.preflight_prepared_dialogue(_dialogue(turn), _config(tmp_path))
    assert calls == []


@pytest.mark.parametrize(
    ("rate", "factor"), [("slow", "0.85"), ("normal", None), ("fast", "1.15")]
)
def test_prepared_rate_postprocess_is_after_worker_and_forces_pcm16(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    rate: str,
    factor: str | None,
) -> None:
    turn = _prepared_turn(
        tmp_path,
        rate=rate,
        pause_before="long",
        pause_within=0,
        hesitations=0,
    )
    captured_requests: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    monkeypatch.setattr(higgs, "_get_worker", lambda *args: object())
    monkeypatch.setattr(higgs, "_request", _successful_request(captured_requests))

    def fake_ffmpeg(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        assert Path(command[3]).is_file(), "worker raw output must exist first"
        commands.append(command)
        Path(command[-1]).write_bytes(Path(command[3]).read_bytes())
        return subprocess.CompletedProcess(command, 0, "", "")

    if factor is None:
        monkeypatch.setattr(
            higgs.subprocess,
            "run",
            lambda *args, **kwargs: pytest.fail("normal rate must not invoke FFmpeg"),
        )
    else:
        monkeypatch.setattr(higgs.subprocess, "run", fake_ffmpeg)

    dialogue = _dialogue(turn)
    config = _config(tmp_path)
    tts_engine.preflight_prepared_dialogue(dialogue, config)
    clock = iter(range(8))
    monkeypatch.setattr(higgs.time, "perf_counter", lambda: next(clock))
    caplog.set_level(logging.INFO, logger="tts5703.tts_engine")
    result = asyncio.run(
        tts_engine.synthesize_prepared_turns(dialogue, tmp_path, config)
    )[0]

    assert len(captured_requests) == 1
    if factor is None:
        assert commands == []
        assert result.rate_status == "not_required"
    else:
        assert commands == [
            [
                "/fake/ffmpeg",
                "-y",
                "-i",
                str(tmp_path / "turn_001.higgs_raw.wav"),
                "-af",
                f"atempo={factor}",
                "-c:a",
                "pcm_s16le",
                str(tmp_path / "turn_001.higgs_processed.part.wav"),
            ]
        ]
        assert result.rate_status == "executed"
    with wave.open(str(result.output_path), "rb") as output:
        assert output.getsampwidth() == 2
        assert output.getnframes() / output.getframerate() == pytest.approx(0.2)
    assert not (tmp_path / "turn_001.higgs_raw.wav").exists()
    timing_logs = [
        record.message
        for record in caplog.records
        if "event=higgs_turn_timing" in record.message
    ]
    assert timing_logs == [
        (
            "event=higgs_turn_timing ordinal=1 speaker=spk_001 "
            f"rate={rate} request_sec=1.000 validation_sec=1.000 "
            "rate_postprocess_sec=1.000 total_sec=7.000"
        )
    ]
