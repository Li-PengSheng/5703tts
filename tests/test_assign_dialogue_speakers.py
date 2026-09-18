"""Offline tests for deterministic final-dialogue speaker assignment."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "assign_dialogue_speakers.py"
SPEC = importlib.util.spec_from_file_location("assign_dialogue_speakers", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
assign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = assign
SPEC.loader.exec_module(assign)

ACTIVE_PATH = ROOT / "data" / "speaker_pool" / "vctk_v0.1" / "active_speakers.json"
REGISTRY_PATH = ROOT / "data" / "speaker_pool" / "vctk_v0.1" / "speaker_registry.json"
EXPECTED_ACTIVE = (
    "spk_001",
    "spk_003",
    "spk_004",
    "spk_006",
    "spk_007",
    "spk_008",
    "spk_009",
    "spk_010",
    "spk_011",
    "spk_012",
)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _dialogue(dialogue_id: str) -> dict[str, Any]:
    required = {
        "rate": "normal",
        "arousal": 2,
        "affect": "neutral",
        "pause_before": "none",
        "pause_within": 0,
        "hesitations": 0,
    }
    return {
        "schema_version": "1.0",
        "dialogue_id": dialogue_id,
        "scenario": {
            "speakers": {
                "caller": {"speaker_id": f"C-{dialogue_id}"},
                "counsellor": {"speaker_id": f"L-{dialogue_id}"},
            }
        },
        "turns": [
            {
                "turn_id": "u1",
                "speaker": "User",
                "text": "I need help.",
                "labels": {},
                "acoustic": {"required": required, "best_effort": {}},
            },
            {
                "turn_id": "l1",
                "speaker": "Listener",
                "text": "I am here.",
                "labels": {},
                "acoustic": {
                    "required": {**required, "rate": "slow", "arousal": 1},
                    "best_effort": {},
                },
            },
        ],
    }


def _pool(tmp_path: Path) -> tuple[Path, Path]:
    speaker_ids = ("spk_001", "spk_002", "spk_003", "spk_004")
    registry = _write_json(
        tmp_path / "registry.json",
        {
            "speaker_pool_version": "test_pool",
            "speakers": [{"speaker_id": item} for item in speaker_ids],
        },
    )
    active = _write_json(
        tmp_path / "active.json",
        {
            "speaker_pool_version": "test_pool",
            "status": "active",
            "selection_basis": "test",
            "active_speaker_ids": list(speaker_ids),
            "excluded_from_active_pool": [],
        },
    )
    return active, registry


def _run(tmp_path: Path, input_path: Path, *, seed: int = 5703):
    active, registry = _pool(tmp_path)
    output = tmp_path / "assignments.jsonl"
    summary = tmp_path / "summary.json"
    result = assign.run_assignment(
        input_path=input_path,
        active_speakers_path=active,
        registry_path=registry,
        output_path=output,
        summary_path=summary,
        seed=seed,
    )
    return *result, output, summary


def test_committed_active_pool_is_frozen_and_valid() -> None:
    pool = assign.load_and_validate_active_pool(ACTIVE_PATH, REGISTRY_PATH)

    assert pool.active_speaker_ids == EXPECTED_ACTIVE
    assert len(pool.active_speaker_ids) == 10
    assert {item.speaker_id for item in pool.exclusions} == {"spk_002", "spk_005"}


def test_final_json_assignment_is_deterministic_and_source_immutable(
    tmp_path: Path,
) -> None:
    source = _write_json(tmp_path / "input" / "dialogue.json", _dialogue("D001"))
    before = source.read_bytes()

    first = _run(tmp_path / "one", source)
    second = _run(tmp_path / "two", source)

    assert first[2].read_bytes() == second[2].read_bytes()
    assert first[3].read_bytes() == second[3].read_bytes()
    assert source.read_bytes() == before
    roles = first[0][0]["role_assignments"]
    assert set(roles) == {"caller", "counsellor"}
    assert len(set(roles.values())) == 2
    assert all(speaker.startswith("spk_") for speaker in roles.values())


def test_final_jsonl_records_are_all_assigned(tmp_path: Path) -> None:
    source = tmp_path / "dialogues.jsonl"
    source.write_text(
        "\n".join(json.dumps(_dialogue(item)) for item in ("D002", "D001")) + "\n",
        encoding="utf-8",
    )

    assignments, summary, _, _ = _run(tmp_path / "run", source)

    assert [item["dialogue_id"] for item in assignments] == ["D001", "D002"]
    assert summary["input_dialogue_count"] == 2
    assert summary["acoustic_exposure"]


def test_assignment_uses_only_final_contract(tmp_path: Path) -> None:
    source = _write_json(
        tmp_path / "old.json",
        {
            "dialogue_id": "old",
            "turns": [{"turn_id": 1, "speaker": "caller", "text": "old"}],
        },
    )

    with pytest.raises(assign.SpeakerAssignmentError, match="scenario must"):
        _run(tmp_path / "run", source)


def test_duplicate_dialogue_ids_fail(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    _write_json(input_dir / "a.json", _dialogue("duplicate"))
    _write_json(input_dir / "b.json", _dialogue("duplicate"))

    with pytest.raises(assign.SpeakerAssignmentError, match="Duplicate dialogue_id"):
        _run(tmp_path / "run", input_dir)


def test_assignments_do_not_depend_on_acoustic_values(tmp_path: Path) -> None:
    first = _dialogue("D001")
    second = _dialogue("D001")
    second["turns"][0]["acoustic"]["required"].update(
        {"rate": "fast", "arousal": 3, "affect": "angry"}
    )
    one = _write_json(tmp_path / "one.json", first)
    two = _write_json(tmp_path / "two.json", second)

    first_assignments = _run(tmp_path / "run-one", one)[0]
    second_assignments = _run(tmp_path / "run-two", two)[0]

    assert first_assignments == second_assignments


def test_invalid_active_pool_is_rejected(tmp_path: Path) -> None:
    active, registry = _pool(tmp_path)
    payload = json.loads(active.read_text(encoding="utf-8"))
    payload["active_speaker_ids"].append("missing")
    _write_json(active, payload)

    with pytest.raises(
        assign.SpeakerAssignmentError, match="not in the speaker registry"
    ):
        assign.load_and_validate_active_pool(active, registry)
