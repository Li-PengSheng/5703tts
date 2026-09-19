from __future__ import annotations

import asyncio
import hashlib
import wave
from pathlib import Path
from typing import Any

import pytest

from tts5703 import render_plan, tts_engine
from tts5703.backends import cosyvoice, higgs
from tts5703.backends.info import describe_engine
from tts5703.input.records import InputRecord
from tts5703.render.assemble import assemble_prepared_dialogue
from tts5703.render.metadata import build_metadata
from tts5703.render.qc import run_qc
from tts5703.render_models import CanonicalTurn, CosyVoicePreparedTurn, TurnRenderResult
from tts5703.render_plan import (
    RenderPlanError,
    canonicalize_dialogue,
    prepare_dialogue,
)


def _wav(path: Path, duration_ms: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8_000)
        output.writeframes(b"\xe8\x03" * (8_000 * duration_ms // 1_000))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    tmp_path: Path, *, pause_within: int = 0
) -> tuple[InputRecord, dict[str, Any]]:
    raw = {
        "schema_version": "1.0",
        "dialogue_id": "dual-final",
        "scenario": {
            "speakers": {
                "caller": {"speaker_id": "C-source"},
                "counsellor": {"speaker_id": "L-source"},
            }
        },
        "turns": [
            {
                "turn_id": "arbitrary-id",
                "speaker": "User",
                "text": "I need a moment. Please stay.",
                "labels": {"opaque": [1, "x"]},
                "acoustic": {
                    "required": {
                        "rate": "slow",
                        "arousal": 3,
                        "affect": "anxious",
                        "pause_before": "short",
                        "pause_within": pause_within,
                        "hesitations": 1,
                    },
                    "best_effort": {
                        "affect_fine": "worried",
                        "volume": None,
                        "flattened_affect": False,
                        "events": [],
                    },
                },
            }
        ],
    }
    record = InputRecord(
        container_path=tmp_path / "input.jsonl",
        source_format="jsonl",
        line_number=1,
        raw=raw,
        dialogue_id=raw["dialogue_id"],
    )
    higgs_ref = tmp_path / "refs" / "higgs.wav"
    cosy_ref = tmp_path / "refs" / "cosy.wav"
    _wav(higgs_ref)
    _wav(cosy_ref)
    role = {
        "upstream_role": "User",
        "upstream_scenario_speaker_id": "C-source",
        "render_speaker_id": "spk_caller",
        "higgs_reference": {
            "reference_wav": "refs/higgs.wav",
            "sha256": _sha(higgs_ref),
        },
        "cosyvoice_reference": {
            "prompt_wav": "refs/cosy.wav",
            "prompt_text": "Exact calibrated prompt.<|endofprompt|>",
            "sha256": _sha(cosy_ref),
        },
    }
    sidecar = {
        "assignment_sha256": "a" * 64,
        "registry_sha256": "b" * 64,
        "active_speakers_sha256": "c" * 64,
        "dialogues": [
            {
                "dialogue_id": record.dialogue_id,
                "source": {"record_sha256": record.record_sha256},
                "roles": {
                    "caller": role,
                    "counsellor": {
                        **role,
                        "upstream_role": "Listener",
                        "upstream_scenario_speaker_id": "L-source",
                        "render_speaker_id": "spk_counsellor",
                    },
                },
            }
        ],
    }
    return record, sidecar


def _config(engine: str) -> dict[str, Any]:
    return {
        "tts": {
            "engine": engine,
            "higgs": {
                "server_executable": "bin/sgl-omni",
                "model_dir": "models/higgs",
                "ffmpeg_bin": "ffmpeg",
            },
            "cosyvoice": {
                "python_bin": "third_party/CosyVoice/.venv/bin/python",
                "repo_dir": "third_party/CosyVoice",
                "model_dir": "models/Fun-CosyVoice3-0.5B",
            },
        },
        "fade_ms": 1,
    }


def test_same_final_record_builds_backend_neutral_then_both_backend_plans(
    tmp_path: Path,
) -> None:
    record, sidecar = _fixture(tmp_path)
    canonical = canonicalize_dialogue(record, sidecar)
    higgs_dialogue = prepare_dialogue(
        record, sidecar, _config("higgs"), project_root=tmp_path
    )
    cosy_dialogue = prepare_dialogue(
        record, sidecar, _config("cosyvoice"), project_root=tmp_path
    )

    assert isinstance(canonical.turns[0], CanonicalTurn)
    assert canonical.turns[0].__dict__ == {
        "ordinal": 1,
        "source_turn_id": "arbitrary-id",
        "text": "I need a moment. Please stay.",
        "labels": {"opaque": [1, "x"]},
        "upstream_role": "User",
        "logical_role": "caller",
        "upstream_scenario_speaker_id": "C-source",
        "render_speaker_id": "spk_caller",
        "required": record.raw["turns"][0]["acoustic"]["required"],
        "best_effort": record.raw["turns"][0]["acoustic"]["best_effort"],
        "rate": "slow",
        "arousal": "high",
        "affect": "anxious",
        "pause_before": "short",
        "pause_before_ms": 500,
        "pause_within_count": 0,
        "hesitation_count": 1,
    }
    assert "model_input" not in str(canonical)
    assert higgs_dialogue.turns[0].plan["higgs"]["model_input"]
    assert higgs_dialogue.turns[0].pause_before_ms == 500
    cosy_turn = cosy_dialogue.turns[0]
    assert isinstance(cosy_turn, CosyVoicePreparedTurn)
    assert "higgs" not in cosy_turn.plan
    assert cosy_turn.plan["normalized"]["arousal"] == "high"
    assert cosy_turn.plan["normalized"]["coarse_affect"] == "anxious"
    assert cosy_turn.plan["cosyvoice"]["speed"] == 0.8
    assert cosy_turn.plan["cosyvoice"]["text"].startswith("I need a moment.")
    assert "um" in cosy_turn.plan["cosyvoice"]["text"]
    assert cosy_turn.render_speaker_id == "spk_caller"
    assert cosy_turn.prompt_text == "Exact calibrated prompt.<|endofprompt|>"
    assert cosy_turn.pause_before_ms == 500


@pytest.mark.parametrize(
    ("engine", "selected", "unselected"),
    [
        ("higgs", "higgs_reference", "cosyvoice_reference"),
        ("cosyvoice", "cosyvoice_reference", "higgs_reference"),
    ],
)
def test_only_selected_backend_reference_is_required(
    tmp_path: Path, engine: str, selected: str, unselected: str
) -> None:
    record, sidecar = _fixture(tmp_path)
    role = sidecar["dialogues"][0]["roles"]["caller"]
    role.pop(unselected)
    prepare_dialogue(record, sidecar, _config(engine), project_root=tmp_path)

    record, sidecar = _fixture(tmp_path / "missing")
    sidecar["dialogues"][0]["roles"]["caller"].pop(selected)
    with pytest.raises(RenderPlanError, match=selected):
        prepare_dialogue(
            record, sidecar, _config(engine), project_root=tmp_path / "missing"
        )


def test_cosyvoice_prompt_text_is_preserved_byte_for_byte(tmp_path: Path) -> None:
    record, sidecar = _fixture(tmp_path)
    prompt_text = "  Exact registry prompt.  \n"
    sidecar["dialogues"][0]["roles"]["caller"]["cosyvoice_reference"]["prompt_text"] = (
        prompt_text
    )

    dialogue = prepare_dialogue(
        record, sidecar, _config("cosyvoice"), project_root=tmp_path
    )

    assert dialogue.turns[0].prompt_text == prompt_text
    assert dialogue.turns[0].plan["cosyvoice"]["prompt_text"] == prompt_text


@pytest.mark.parametrize(
    ("pause_within", "break_plan"),
    [
        (0, lambda plan: plan["cosyvoice"].__setitem__("speed", 1.2)),
        (
            0,
            lambda plan: plan["realization"]["pause_within_count"].__setitem__(
                "status", "unsupported"
            ),
        ),
        (
            0,
            lambda plan: plan["capabilities"].__setitem__(
                "pause_within", "silently_ignored"
            ),
        ),
        (
            1,
            lambda plan: plan["realization"]["pause_within_count"].__setitem__(
                "status", "not_required"
            ),
        ),
    ],
)
def test_cosyvoice_cached_plan_rejects_rate_and_pause_capability_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pause_within: int,
    break_plan: Any,
) -> None:
    record, sidecar = _fixture(tmp_path, pause_within=pause_within)
    planner = render_plan._cosyvoice_plan

    def broken_plan(turn: CanonicalTurn, reference: dict[str, Any]) -> dict[str, Any]:
        plan = planner(turn, reference)
        break_plan(plan)
        return plan

    monkeypatch.setattr(render_plan, "_cosyvoice_plan", broken_plan)

    with pytest.raises(RenderPlanError):
        prepare_dialogue(record, sidecar, _config("cosyvoice"), project_root=tmp_path)


def test_cosyvoice_pause_within_is_explicit_and_fails_closed_before_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, sidecar = _fixture(tmp_path, pause_within=1)
    dialogue = prepare_dialogue(
        record, sidecar, _config("cosyvoice"), project_root=tmp_path
    )
    plan = dialogue.turns[0].plan
    calls: list[str] = []
    monkeypatch.setattr(
        cosyvoice, "_runtime", lambda _: ({}, tmp_path, tmp_path, tmp_path)
    )
    monkeypatch.setattr(
        cosyvoice.cosyvoice, "_get_worker", lambda *args: calls.append("worker")
    )

    assert plan["capabilities"]["pause_within"] == "fail_closed_when_requested"
    assert plan["realization"]["pause_within_count"]["status"] == "unsupported"
    with pytest.raises(RuntimeError, match="no evidence-backed deterministic"):
        tts_engine.preflight_prepared_dialogue(dialogue, _config("cosyvoice"))
    assert calls == []
    assert not (tmp_path / "output").exists()


def test_cosyvoice_preflight_rechecks_prompt_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, sidecar = _fixture(tmp_path)
    dialogue = prepare_dialogue(
        record, sidecar, _config("cosyvoice"), project_root=tmp_path
    )
    dialogue.turns[0].resolved_prompt_wav.write_bytes(b"changed")
    monkeypatch.setattr(
        cosyvoice, "_runtime", lambda _: ({}, tmp_path, tmp_path, tmp_path)
    )

    with pytest.raises(RuntimeError, match="prompt SHA-256 mismatch"):
        tts_engine.preflight_prepared_dialogue(dialogue, _config("cosyvoice"))


def test_cosyvoice_execution_uses_cached_request_and_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, sidecar = _fixture(tmp_path)
    dialogue = prepare_dialogue(
        record, sidecar, _config("cosyvoice"), project_root=tmp_path
    )
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cosyvoice,
        "_runtime",
        lambda _: ({"fp16": False}, tmp_path, tmp_path, tmp_path),
    )
    monkeypatch.setattr(cosyvoice.cosyvoice, "_get_worker", lambda *args: object())

    def request(_worker: object, payload: dict[str, Any]) -> dict[str, str]:
        captured.append(payload)
        _wav(Path(payload["output_path"]))
        return {"status": "ok", "output_path": payload["output_path"]}

    monkeypatch.setattr(cosyvoice.cosyvoice, "_request", request)
    monkeypatch.setattr(
        higgs,
        "synthesize_prepared_turn",
        lambda *args: (_ for _ in ()).throw(AssertionError("automatic fallback")),
    )

    tts_engine.preflight_prepared_dialogue(dialogue, _config("cosyvoice"))
    results = asyncio.run(
        tts_engine.synthesize_prepared_turns(
            dialogue, tmp_path / "output", _config("cosyvoice")
        )
    )
    plan = dialogue.turns[0].plan["cosyvoice"]
    assert results[0].ordinal == 1
    assert results[0].source_turn_id == "arbitrary-id"
    assert results[0].rate_status == "executed"
    assert captured[0]["text"] == plan["text"]
    assert captured[0]["speed"] == plan["speed"]
    assert captured[0]["instruction"] == plan["instruction"]
    assert captured[0]["prompt_text"] == plan["prompt_text"]


def test_cosyvoice_failure_is_not_retried_with_higgs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, sidecar = _fixture(tmp_path)
    dialogue = prepare_dialogue(
        record, sidecar, _config("cosyvoice"), project_root=tmp_path
    )
    fallback_calls: list[str] = []
    monkeypatch.setattr(
        cosyvoice,
        "_runtime",
        lambda _: ({}, tmp_path, tmp_path, tmp_path),
    )
    monkeypatch.setattr(cosyvoice.cosyvoice, "_get_worker", lambda *args: object())
    monkeypatch.setattr(
        cosyvoice.cosyvoice,
        "_request",
        lambda *args: (_ for _ in ()).throw(RuntimeError("cosy failed")),
    )
    monkeypatch.setattr(
        higgs,
        "synthesize_prepared_turn",
        lambda *args: fallback_calls.append("higgs"),
    )

    tts_engine.preflight_prepared_dialogue(dialogue, _config("cosyvoice"))
    with pytest.raises(RuntimeError, match="cosy failed"):
        asyncio.run(
            tts_engine.synthesize_prepared_turns(
                dialogue, tmp_path / "output", _config("cosyvoice")
            )
        )
    assert fallback_calls == []


def test_cosyvoice_metadata_and_qc_use_cached_plan_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, sidecar = _fixture(tmp_path)
    config = _config("cosyvoice")
    dialogue = prepare_dialogue(record, sidecar, config, project_root=tmp_path)
    turn_audio = tmp_path / "turn_001.wav"
    _wav(turn_audio)
    results = (
        TurnRenderResult(1, "arbitrary-id", turn_audio, "synthesized", "executed"),
    )
    audio, timings = assemble_prepared_dialogue(dialogue, results, config)
    clean, telephone = tmp_path / "clean.wav", tmp_path / "telephone.wav"
    audio.export(clean, format="wav")
    audio.export(telephone, format="wav")
    metadata = build_metadata(
        record,
        dialogue,
        clean,
        telephone,
        timings,
        results,
        describe_engine(config, dialogue),
    )
    monkeypatch.setattr(
        render_plan,
        "plan_hesitations",
        lambda *args: (_ for _ in ()).throw(AssertionError("planner recalled")),
    )
    monkeypatch.setattr(
        render_plan,
        "map_turn_to_higgs",
        lambda *args: (_ for _ in ()).throw(AssertionError("planner recalled")),
    )

    assert metadata["tts"]["engine"] == "cosyvoice"
    assert metadata["tts"]["backend_identity"]["text_frontend"] is False
    assert metadata["turns"][0]["planned"] == dialogue.turns[0].plan
    assert "higgs" not in str(metadata["turns"][0]["planned"])
    assert metadata["turns"][0]["approved_speaker_reference"]["prompt_text"]
    assert run_qc(record, dialogue, results, timings, clean, telephone, metadata).passed


def test_new_final_cosyvoice_backend_has_no_legacy_or_higgs_imports() -> None:
    source = "\n".join(
        (
            Path(cosyvoice.__file__).read_text(encoding="utf-8"),
            Path(cosyvoice.cosyvoice.__file__).read_text(encoding="utf-8"),
            Path(render_plan.__file__).read_text(encoding="utf-8"),
        )
    )
    for forbidden in (
        "NormalizedTurn",
        "load_and_validate",
        "schemas/dialogue_schema.json",
        "higgs_controls",
    ):
        assert forbidden not in source
