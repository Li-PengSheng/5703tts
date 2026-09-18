from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tts5703 import render_plan
from tts5703.input_records import InputRecord
from tts5703.render_models import (
    CanonicalTurn,
    CosyVoicePreparedTurn,
    HiggsPreparedTurn,
)
from tts5703.render_plan import RenderPlanError, prepare_dialogue


def _required(
    *,
    rate: str = "normal",
    arousal: int = 2,
    affect: str = "neutral",
    pause_before: str = "none",
    pause_within: int = 0,
    hesitations: int = 0,
) -> dict[str, Any]:
    return {
        "rate": rate,
        "arousal": arousal,
        "affect": affect,
        "pause_before": pause_before,
        "pause_within": pause_within,
        "hesitations": hesitations,
    }


def _raw() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "dialogue_id": "final-001",
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
                "labels": {"opaque": ["one"]},
                "acoustic": {
                    "required": _required(
                        rate="slow",
                        arousal=1,
                        affect="warm",
                        pause_before="short",
                        pause_within=2,
                        hesitations=2,
                    ),
                    "best_effort": {},
                },
            },
            {
                "turn_id": "listener-zeta",
                "speaker": "Listener",
                "text": "I am here with you.",
                "labels": ["opaque"],
                "acoustic": {
                    "required": _required(rate="fast", pause_before="long"),
                    "best_effort": {},
                },
            },
        ],
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> tuple[InputRecord, dict[str, Any]]:
    raw = _raw()
    record = InputRecord(
        container_path=tmp_path / "corpus.jsonl",
        source_format="jsonl",
        line_number=7,
        raw=raw,
        dialogue_id=raw["dialogue_id"],
    )
    caller_reference = tmp_path / "refs" / "caller.wav"
    listener_reference = tmp_path / "refs" / "listener.wav"
    caller_reference.parent.mkdir()
    caller_reference.write_bytes(b"approved caller reference")
    listener_reference.write_bytes(b"approved listener reference")
    sidecar = {
        "assignment_sha256": "a" * 64,
        "registry_sha256": "b" * 64,
        "active_speakers_sha256": "c" * 64,
        "dialogues": [
            {
                "dialogue_id": raw["dialogue_id"],
                "source": {
                    "container_path": "old/location.jsonl",
                    "format": "jsonl",
                    "line_number": 999,
                    "record_sha256": record.record_sha256,
                },
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
                        "render_speaker_id": "spk_003",
                        "higgs_reference": {
                            "reference_wav": "refs/listener.wav",
                            "sha256": _sha(listener_reference),
                        },
                    },
                },
            }
        ],
    }
    return record, sidecar


def _direct_higgs_turn(tmp_path: Path, plan: dict[str, Any]) -> HiggsPreparedTurn:
    return HiggsPreparedTurn(
        ordinal=1,
        source_turn_id="caller-alpha",
        upstream_role="User",
        logical_role="caller",
        upstream_scenario_speaker_id="C123",
        render_speaker_id="spk_001",
        reference_wav="refs/caller.wav",
        resolved_reference_wav=tmp_path / "refs" / "caller.wav",
        reference_sha256="a" * 64,
        plan=plan,
    )


def _direct_cosyvoice_turn(
    tmp_path: Path, turn: CanonicalTurn, plan: dict[str, Any]
) -> CosyVoicePreparedTurn:
    return CosyVoicePreparedTurn(
        ordinal=turn.ordinal,
        source_turn_id=turn.source_turn_id,
        upstream_role=turn.upstream_role,
        logical_role=turn.logical_role,
        upstream_scenario_speaker_id=turn.upstream_scenario_speaker_id,
        render_speaker_id=turn.render_speaker_id,
        prompt_wav="refs/prompt.wav",
        resolved_prompt_wav=tmp_path / "refs" / "prompt.wav",
        prompt_text="Exact prompt",
        reference_sha256="b" * 64,
        plan=plan,
    )


def _cosyvoice_turn_and_plan() -> tuple[CanonicalTurn, dict[str, Any]]:
    required = _required(rate="slow", arousal=3, affect="anxious")
    turn = CanonicalTurn(
        ordinal=1,
        source_turn_id="caller-alpha",
        text="Please stay with me.",
        labels={},
        upstream_role="User",
        logical_role="caller",
        upstream_scenario_speaker_id="C123",
        render_speaker_id="spk_001",
        required=required,
        best_effort={},
        rate="slow",
        arousal="high",
        affect="anxious",
        pause_before="none",
        pause_before_ms=0,
        pause_within_count=0,
        hesitation_count=0,
    )
    plan = render_plan._cosyvoice_plan(
        turn,
        {
            "prompt_wav": "refs/prompt.wav",
            "prompt_text": "Exact prompt",
            "sha256": "b" * 64,
        },
    )
    return turn, plan


def test_direct_higgs_prepared_turn_accepts_valid_plan(tmp_path: Path) -> None:
    raw = _raw()
    plan = render_plan.map_turn_to_higgs(raw["turns"][0], dialogue_context=raw)

    prepared = _direct_higgs_turn(tmp_path, plan)

    assert prepared.plan == plan


def test_direct_higgs_prepared_turn_rejects_invalid_plan(tmp_path: Path) -> None:
    raw = _raw()
    plan = render_plan.map_turn_to_higgs(raw["turns"][0], dialogue_context=raw)
    plan["postprocess"]["atempo"]["processor"] = "not_ffmpeg"

    with pytest.raises(RenderPlanError, match="rate plan is inconsistent"):
        _direct_higgs_turn(tmp_path, plan)


def test_direct_cosyvoice_prepared_turn_accepts_valid_plan(tmp_path: Path) -> None:
    turn, plan = _cosyvoice_turn_and_plan()

    prepared = _direct_cosyvoice_turn(tmp_path, turn, plan)

    assert prepared.plan == plan


def test_direct_cosyvoice_prepared_turn_rejects_invalid_plan(
    tmp_path: Path,
) -> None:
    turn, plan = _cosyvoice_turn_and_plan()
    plan["cosyvoice"]["speed"] = 1.2

    with pytest.raises(RenderPlanError, match="rate and speed plan is inconsistent"):
        _direct_cosyvoice_turn(tmp_path, turn, plan)


def test_preparation_maps_once_per_turn_and_preserves_complete_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, sidecar = _inputs(tmp_path)
    original_record = record.raw
    original_sidecar = deepcopy(sidecar)
    mapper = render_plan.map_turn_to_higgs
    mapped: list[dict[str, Any]] = []

    def counting_mapper(
        turn: dict[str, Any], *, dialogue_context: dict[str, Any]
    ) -> dict[str, Any]:
        plan = mapper(turn, dialogue_context=dialogue_context)
        mapped.append(deepcopy(plan))
        return plan

    monkeypatch.setattr(render_plan, "map_turn_to_higgs", counting_mapper)
    prepared = prepare_dialogue(record, sidecar, project_root=tmp_path)

    assert len(mapped) == len(record.raw["turns"]) == 2
    assert [turn.plan for turn in prepared.turns] == mapped
    assert record.raw == original_record
    assert sidecar == original_sidecar
    assert prepared.assignment_sha256 == "a" * 64
    assert prepared.registry_sha256 == "b" * 64
    assert prepared.active_speakers_sha256 == "c" * 64


def test_prepared_plan_is_deeply_isolated_from_callers(tmp_path: Path) -> None:
    record, sidecar = _inputs(tmp_path)
    turn = prepare_dialogue(record, sidecar, project_root=tmp_path).turns[0]
    expected = turn.plan

    exposed = turn.plan
    exposed["higgs"]["model_input"] = "mutated"
    exposed["normalized"]["rate"] = "fast"
    exposed["postprocess"]["pause_before_ms"] = 999
    exposed["planner"]["pause_within"]["events"].clear()

    assert turn.plan == expected
    assert turn.model_input == expected["higgs"]["model_input"]
    assert turn.rate == "slow"
    assert turn.pause_before_ms == 500
    assert turn.native_pause_token_count == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda record, sidecar: sidecar["dialogues"][0]["source"].__setitem__(
                "record_sha256", "0" * 64
            ),
            "record_sha256",
        ),
        (
            lambda record, sidecar: sidecar["dialogues"][0].__setitem__(
                "dialogue_id", "different"
            ),
            "dialogue_id",
        ),
        (
            lambda record, sidecar: sidecar["dialogues"][0]["roles"].pop("caller"),
            "roles",
        ),
        (
            lambda record, sidecar: sidecar["dialogues"][0]["roles"][
                "caller"
            ].__setitem__("upstream_role", "Listener"),
            "upstream_role",
        ),
        (
            lambda record, sidecar: sidecar["dialogues"][0]["roles"]["caller"].pop(
                "higgs_reference"
            ),
            "higgs_reference",
        ),
        (
            lambda record, sidecar: sidecar["dialogues"][0]["roles"]["caller"][
                "higgs_reference"
            ].__setitem__("sha256", "bad"),
            "64 hexadecimal",
        ),
    ],
)
def test_stale_or_incomplete_sidecar_is_rejected(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    record, sidecar = _inputs(tmp_path)
    mutation(record, sidecar)

    with pytest.raises(RenderPlanError, match=message):
        prepare_dialogue(record, sidecar, project_root=tmp_path)


def test_missing_reference_file_is_rejected(tmp_path: Path) -> None:
    record, sidecar = _inputs(tmp_path)
    (tmp_path / "refs" / "caller.wav").unlink()

    with pytest.raises(RenderPlanError, match="reference file is missing"):
        prepare_dialogue(record, sidecar, project_root=tmp_path)


def test_reference_sha_mismatch_is_rejected(tmp_path: Path) -> None:
    record, sidecar = _inputs(tmp_path)
    (tmp_path / "refs" / "caller.wav").write_bytes(b"changed")

    with pytest.raises(RenderPlanError, match="SHA-256 mismatch"):
        prepare_dialogue(record, sidecar, project_root=tmp_path)


def test_speaker_identities_remain_distinct(tmp_path: Path) -> None:
    record, sidecar = _inputs(tmp_path)
    prepared = prepare_dialogue(record, sidecar, project_root=tmp_path)

    caller = prepared.turns[0]
    assert caller.upstream_role == "User"
    assert caller.logical_role == "caller"
    assert caller.upstream_scenario_speaker_id == "C123"
    assert caller.render_speaker_id == "spk_001"
    assert caller.plan["logical_speaker"]["speaker_id"] == "C123"
    assert caller.render_speaker_id != caller.upstream_scenario_speaker_id


def test_container_path_and_line_move_do_not_invalidate_record_identity(
    tmp_path: Path,
) -> None:
    record, sidecar = _inputs(tmp_path)
    moved = InputRecord(
        container_path=tmp_path / "moved" / "different.jsonl",
        source_format="jsonl",
        line_number=101,
        raw=record.raw,
        dialogue_id=record.dialogue_id,
    )

    prepared = prepare_dialogue(moved, sidecar, project_root=tmp_path)
    assert moved.record_sha256 == record.record_sha256 == prepared.record_sha256


@pytest.mark.parametrize(
    "break_plan",
    [
        lambda plan: plan["higgs"].__setitem__("model_input", ""),
        lambda plan: plan["higgs"].__setitem__("synthesis_call_count", 2),
        lambda plan: plan["postprocess"].__setitem__("atempo_factor", 9.9),
        lambda plan: plan["postprocess"]["atempo"].__setitem__(
            "processor", "not_ffmpeg"
        ),
        lambda plan: plan["postprocess"].__setitem__("pause_before_ms", 1),
        lambda plan: plan["higgs"].__setitem__("native_pause_token_count", 99),
    ],
)
def test_impossible_frozen_execution_shape_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, break_plan: Any
) -> None:
    record, sidecar = _inputs(tmp_path)
    mapper = render_plan.map_turn_to_higgs

    def broken_mapper(
        turn: dict[str, Any], *, dialogue_context: dict[str, Any]
    ) -> dict[str, Any]:
        plan = mapper(turn, dialogue_context=dialogue_context)
        break_plan(plan)
        return plan

    monkeypatch.setattr(render_plan, "map_turn_to_higgs", broken_mapper)
    with pytest.raises(RenderPlanError):
        prepare_dialogue(record, sidecar, project_root=tmp_path)
