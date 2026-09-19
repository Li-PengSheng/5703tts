from __future__ import annotations

import asyncio
import hashlib
import wave
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tts5703 import pipeline, render_plan
from tts5703.assemble import assemble_prepared_dialogue
from tts5703.backends.capabilities import controlled_tts_v1_capabilities
from tts5703.backends.higgs_worker import FROZEN_GENERATION_FIELDS
from tts5703.backends.info import (
    CONTROLLED_TTS_V1_IMPLEMENTATION_ID,
    backend_identity,
    describe_engine,
)
from tts5703.input_records import InputRecord
from tts5703.metadata import build_metadata
from tts5703.postprocess import apply_telephone_effect
from tts5703.qc import run_qc
from tts5703.render_models import TurnRenderResult
from tts5703.render_plan import prepare_dialogue


def _write_wav(path: Path, duration_ms: int = 100) -> None:
    sample_rate = 8_000
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\xe8\x03" * (sample_rate * duration_ms // 1_000))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required(
    *,
    rate: str = "normal",
    pause_before: str = "none",
    pause_within: int = 0,
    hesitations: int = 0,
) -> dict[str, Any]:
    return {
        "rate": rate,
        "arousal": 2,
        "affect": "neutral",
        "pause_before": pause_before,
        "pause_within": pause_within,
        "hesitations": hesitations,
    }


def _inputs(tmp_path: Path) -> tuple[InputRecord, dict[str, Any]]:
    raw = {
        "schema_version": "1.0",
        "dialogue_id": "final-production-001",
        "scenario": {
            "speakers": {
                "caller": {"speaker_id": "C123"},
                "counsellor": {"speaker_id": "L456"},
            }
        },
        "turns": [
            {
                "turn_id": "caller-alpha",
                "speaker": "User",
                "text": "First sentence. Second sentence.",
                "labels": {"risks": ["opaque", {"nested": True}]},
                "acoustic": {
                    "required": _required(
                        rate="slow",
                        pause_before="short",
                        pause_within=2,
                        hesitations=2,
                    ),
                    "best_effort": {
                        "affect_fine": "hopeful",
                        "volume": "soft",
                        "flattened_affect": False,
                        "events": [{"kind": "sigh"}],
                    },
                },
            },
            {
                "turn_id": "listener-zeta",
                "speaker": "Listener",
                "text": "I am here with you.",
                "labels": ["opaque", 7],
                "acoustic": {
                    "required": _required(pause_before="long"),
                    "best_effort": {},
                },
            },
        ],
    }
    record = InputRecord(
        container_path=tmp_path / "corpus.jsonl",
        source_format="jsonl",
        line_number=12,
        raw=raw,
        dialogue_id=raw["dialogue_id"],
    )
    caller_reference = tmp_path / "refs" / "caller.wav"
    counsellor_reference = tmp_path / "refs" / "counsellor.wav"
    _write_wav(caller_reference, 50)
    _write_wav(counsellor_reference, 50)
    sidecar = {
        "assignment_sha256": "a" * 64,
        "registry_sha256": "b" * 64,
        "active_speakers_sha256": "c" * 64,
        "dialogues": [
            {
                "dialogue_id": record.dialogue_id,
                "source": {"record_sha256": record.record_sha256},
                "roles": {
                    "caller": {
                        "upstream_role": "User",
                        "upstream_scenario_speaker_id": "C123",
                        "render_speaker_id": "spk_001",
                        "higgs_reference": {
                            "reference_wav": "refs/caller.wav",
                            "sha256": _sha(caller_reference),
                        },
                    },
                    "counsellor": {
                        "upstream_role": "Listener",
                        "upstream_scenario_speaker_id": "L456",
                        "render_speaker_id": "spk_002",
                        "higgs_reference": {
                            "reference_wav": "refs/counsellor.wav",
                            "sha256": _sha(counsellor_reference),
                        },
                    },
                },
            }
        ],
    }
    return record, sidecar


def _config(*, engine: str = "higgs") -> dict[str, Any]:
    return {
        "tts": {
            "engine": engine,
            "higgs": {
                "server_executable": "bin/sgl-omni",
                "model_dir": "models/higgs",
                "ffmpeg_bin": "ffmpeg",
            },
        },
        "fade_ms": 5,
        "telephone": {
            "sample_rate": 8_000,
            "channels": 1,
            "high_pass_hz": 300,
            "low_pass_hz": 3_400,
            "volume_db_reduction": 3,
        },
    }


def _artifacts(tmp_path: Path) -> dict[str, Any]:
    record, sidecar = _inputs(tmp_path)
    prepared = prepare_dialogue(record, sidecar, project_root=tmp_path)
    results = []
    for turn in prepared.turns:
        output = tmp_path / f"turn_{turn.ordinal:03d}.wav"
        _write_wav(output)
        results.append(
            TurnRenderResult(
                ordinal=turn.ordinal,
                source_turn_id=turn.source_turn_id,
                output_path=output,
                synthesis_status="synthesized",
                rate_status="executed" if turn.rate != "normal" else "not_required",
            )
        )
    result_tuple = tuple(results)
    audio, timings = assemble_prepared_dialogue(prepared, result_tuple, _config())
    clean = tmp_path / f"{record.dialogue_id}_clean.wav"
    telephone = tmp_path / f"{record.dialogue_id}_telephone.wav"
    audio.export(clean, format="wav")
    apply_telephone_effect(audio, _config()).export(telephone, format="wav")
    engine_info = describe_engine(_config(), prepared)
    plans_before_metadata = [turn.plan for turn in prepared.turns]
    metadata = build_metadata(
        record,
        prepared,
        clean,
        telephone,
        timings,
        result_tuple,
        engine_info,
    )
    return {
        "record": record,
        "prepared": prepared,
        "results": result_tuple,
        "timings": timings,
        "clean": clean,
        "telephone": telephone,
        "metadata": metadata,
        "plans_before_metadata": plans_before_metadata,
    }


def test_final_pipeline_prepares_once_and_orders_preflight_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, sidecar = _inputs(tmp_path)
    output_root = tmp_path / "output"
    dialogue_dir = output_root / record.dialogue_id
    calls: list[str] = []
    mapper_calls = 0
    original_mapper = render_plan.map_turn_to_higgs
    original_prepare = pipeline.prepare_dialogue
    original_assemble = pipeline.assemble_prepared_dialogue
    original_telephone = pipeline.apply_telephone_effect

    def mapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal mapper_calls
        mapper_calls += 1
        return original_mapper(*args, **kwargs)

    def prepare(*args: Any, **kwargs: Any):
        calls.append("prepare")
        return original_prepare(*args, **kwargs)

    def preflight(*args: Any, **kwargs: Any) -> None:
        calls.append("preflight")
        assert not dialogue_dir.exists()

    async def synthesize(dialogue: Any, out_dir: Path, config: dict[str, Any]):
        calls.append("synthesize")
        assert out_dir == dialogue_dir and out_dir.is_dir()
        results = []
        for turn in dialogue.turns:
            path = out_dir / f"turn_{turn.ordinal:03d}.wav"
            _write_wav(path)
            results.append(
                TurnRenderResult(
                    turn.ordinal,
                    turn.source_turn_id,
                    path,
                    "synthesized",
                    "executed" if turn.rate != "normal" else "not_required",
                )
            )
        return tuple(results)

    def assemble(*args: Any, **kwargs: Any):
        calls.append("assemble")
        return original_assemble(*args, **kwargs)

    def telephone(*args: Any, **kwargs: Any):
        calls.append("telephone")
        return original_telephone(*args, **kwargs)

    monkeypatch.setattr(render_plan, "map_turn_to_higgs", mapper)
    monkeypatch.setattr(pipeline, "prepare_dialogue", prepare)
    monkeypatch.setattr(pipeline, "preflight_prepared_dialogue", preflight)
    monkeypatch.setattr(pipeline, "synthesize_prepared_turns", synthesize)
    monkeypatch.setattr(pipeline, "assemble_prepared_dialogue", assemble)
    monkeypatch.setattr(pipeline, "apply_telephone_effect", telephone)

    result = asyncio.run(
        pipeline.run_dialogue(
            record,
            sidecar,
            _config(),
            output_root,
            project_root=tmp_path,
        )
    )

    assert result.status == "success"
    assert calls == ["prepare", "preflight", "synthesize", "assemble", "telephone"]
    assert mapper_calls == len(record.raw["turns"])
    assert result.qc is not None and result.qc.passed


@pytest.mark.parametrize("failure", ["prepare", "preflight"])
def test_final_pipeline_pre_render_failures_leave_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    record, sidecar = _inputs(tmp_path)
    output_root = tmp_path / "output"
    if failure == "prepare":
        sidecar["dialogues"][0]["source"]["record_sha256"] = "0" * 64
    else:
        monkeypatch.setattr(
            pipeline,
            "preflight_prepared_dialogue",
            lambda *args: (_ for _ in ()).throw(RuntimeError("turn 2 invalid")),
        )

    result = asyncio.run(
        pipeline.run_dialogue(
            record,
            sidecar,
            _config(),
            output_root,
            project_root=tmp_path,
        )
    )

    assert result.status == "failed"
    assert not (output_root / record.dialogue_id).exists()


def test_final_metadata_preserves_source_plan_execution_and_provenance(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)
    record = artifacts["record"]
    prepared = artifacts["prepared"]
    metadata = artifacts["metadata"]
    plan_before = prepared.turns[0].plan
    source = record.raw["turns"][0]
    turn = metadata["turns"][0]

    assert [turn.plan for turn in prepared.turns] == artifacts["plans_before_metadata"]

    assert turn["source_identity"] == {
        "ordinal": 1,
        "source_turn_id": "caller-alpha",
        "upstream_role": "User",
        "logical_role": "caller",
        "upstream_scenario_speaker_id": "C123",
        "render_speaker_id": "spk_001",
    }
    assert turn["requested"]["acoustic"]["required"] == source["acoustic"]["required"]
    assert (
        turn["requested"]["acoustic"]["best_effort"]
        == source["acoustic"]["best_effort"]
    )
    assert turn["labels"] == source["labels"]
    assert turn["planned"] == plan_before
    assert (
        turn["approved_speaker_reference"]["sha256"]
        == prepared.turns[0].reference_sha256
    )
    assert turn["execution"] == {
        "synthesis_status": "synthesized",
        "rate_status": "executed",
        "turn_audio": "turn_001.wav",
    }
    assert turn["timing"]["start_sec"] == 0.5
    assert turn["timing"]["pause_before_ms"] == 500
    assert turn["timing"]["pause_after_ms"] == 0
    assert "label" not in turn
    assert metadata["provenance"] == {
        "record_sha256": record.record_sha256,
        "assignment_sha256": "a" * 64,
        "registry_sha256": "b" * 64,
        "active_speakers_sha256": "c" * 64,
        "mapping_version": "controlled_tts_v1",
        "release_status": "PROVISIONAL_PRODUCTION_MAPPING_WEEK6_FREEZE",
        "contract_sha256": (
            "4e787a5363d01e5454a8031c59bb1d4bb454ee37d44ccf262d471c7ce30ba799"
        ),
        "implementation_id": CONTROLLED_TTS_V1_IMPLEMENTATION_ID,
    }
    turn["planned"]["higgs"]["model_input"] = "external mutation"
    assert prepared.turns[0].plan == plan_before


def _corrupt_requested(metadata: dict[str, Any]) -> None:
    metadata["turns"][0]["requested"]["acoustic"]["required"]["rate"] = "fast"


def _corrupt_best_effort(metadata: dict[str, Any]) -> None:
    metadata["turns"][0]["requested"]["acoustic"]["best_effort"]["volume"] = "loud"


def _corrupt_model_input(metadata: dict[str, Any]) -> None:
    metadata["turns"][0]["planned"]["higgs"]["model_input"] = "wrong"


def _corrupt_hesitations(metadata: dict[str, Any]) -> None:
    metadata["turns"][0]["planned"]["planner"]["hesitation"]["inserted_count"] = 99


def _corrupt_native_pauses(metadata: dict[str, Any]) -> None:
    metadata["turns"][0]["planned"]["higgs"]["native_pause_token_count"] = 99


def _corrupt_call_count(metadata: dict[str, Any]) -> None:
    metadata["turns"][0]["planned"]["higgs"]["synthesis_call_count"] = 2


def _corrupt_reference(metadata: dict[str, Any]) -> None:
    metadata["turns"][0]["approved_speaker_reference"]["sha256"] = "0" * 64


def _corrupt_render_speaker(metadata: dict[str, Any]) -> None:
    metadata["turns"][0]["source_identity"]["render_speaker_id"] = "spk_wrong"


def _corrupt_synthesis_status(metadata: dict[str, Any]) -> None:
    metadata["turns"][0]["execution"]["synthesis_status"] = "failed"


def _corrupt_rate_status(metadata: dict[str, Any]) -> None:
    metadata["turns"][0]["execution"]["rate_status"] = "not_required"


def _corrupt_pause_after(metadata: dict[str, Any]) -> None:
    metadata["turns"][0]["timing"]["pause_after_ms"] = 500


def _corrupt_turn_order(metadata: dict[str, Any]) -> None:
    metadata["turns"].reverse()


def _corrupt_timestamp_order(metadata: dict[str, Any]) -> None:
    metadata["turns"][1]["timing"]["start_sec"] = 0.1


@pytest.mark.parametrize(
    "corrupt",
    [
        _corrupt_requested,
        _corrupt_best_effort,
        _corrupt_model_input,
        _corrupt_hesitations,
        _corrupt_native_pauses,
        _corrupt_call_count,
        _corrupt_reference,
        _corrupt_render_speaker,
        _corrupt_synthesis_status,
        _corrupt_rate_status,
        _corrupt_pause_after,
        _corrupt_turn_order,
        _corrupt_timestamp_order,
    ],
)
def test_final_structural_qc_rejects_each_integrity_corruption(
    tmp_path: Path, corrupt: Any
) -> None:
    artifacts = _artifacts(tmp_path)
    metadata = deepcopy(artifacts["metadata"])
    corrupt(metadata)

    result = run_qc(
        artifacts["record"],
        artifacts["prepared"],
        artifacts["results"],
        artifacts["timings"],
        artifacts["clean"],
        artifacts["telephone"],
        metadata,
    )

    assert not result.passed


def test_positive_final_qc_is_explicitly_structural_only(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    result = run_qc(
        artifacts["record"],
        artifacts["prepared"],
        artifacts["results"],
        artifacts["timings"],
        artifacts["clean"],
        artifacts["telephone"],
        artifacts["metadata"],
    )

    assert result.passed
    assert result.checks["scope_structural_control_integrity_only"]
    assert "perceptual" not in str(result).lower()


@pytest.mark.parametrize("artifact", ["turn", "clean", "telephone"])
def test_final_qc_reports_missing_audio_artifacts(
    tmp_path: Path, artifact: str
) -> None:
    artifacts = _artifacts(tmp_path)
    path = {
        "turn": artifacts["results"][0].output_path,
        "clean": artifacts["clean"],
        "telephone": artifacts["telephone"],
    }[artifact]
    path.unlink()

    result = run_qc(
        artifacts["record"],
        artifacts["prepared"],
        artifacts["results"],
        artifacts["timings"],
        artifacts["clean"],
        artifacts["telephone"],
        artifacts["metadata"],
    )

    assert not result.passed


def test_backend_identity_locks_semantics_not_source_bytes(
    tmp_path: Path,
) -> None:
    record, sidecar = _inputs(tmp_path)
    prepared = prepare_dialogue(record, sidecar, project_root=tmp_path)
    identity = backend_identity(_config(), prepared)
    mapping = identity["control_mapping"]

    assert mapping["mapping_version"] == "controlled_tts_v1"
    assert mapping["release_status"] == "PROVISIONAL_PRODUCTION_MAPPING_WEEK6_FREEZE"
    assert mapping["implementation_id"] == "controlled_tts_v1_prod_1"
    assert mapping["provenance"]["contract_sha256"] == (
        "4e787a5363d01e5454a8031c59bb1d4bb454ee37d44ccf262d471c7ce30ba799"
    )
    assert identity["generation_profile"] == FROZEN_GENERATION_FIELDS
    assert (
        identity["references"]["spk_001"]["sha256"]
        == prepared.turns[0].reference_sha256
    )
    assert "source_sha256" not in str(identity)
    assert "higgs_worker.py" not in str(identity)


def test_final_capabilities_are_separate_and_truthful() -> None:
    capabilities = controlled_tts_v1_capabilities()
    assert capabilities["required"]["pause_within"]["realization"] == (
        "native_higgs_pause_token_planner"
    )
    assert capabilities["required"]["hesitations"]["realization"] == ("lexical_planner")
    assert capabilities["best_effort"]["affect_fine"]["support"] == (
        "conditional_joint_realization"
    )
    assert capabilities["best_effort"]["volume"]["support"] == "unsupported"
    assert capabilities["best_effort"]["events"]["possible_statuses"] == [
        "not_realized",
        "unsupported",
    ]

    capabilities["required"].clear()
    assert "pause_within" in controlled_tts_v1_capabilities()["required"]
