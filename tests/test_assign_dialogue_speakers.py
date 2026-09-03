"""Offline tests for the frozen v0.1 active pool and dialogue assignment.

No VCTK audio, CosyVoice, GPU, or network is used. Synthetic pools live in
``tmp_path``; committed JSON is read only for the real v0.1 freeze checks.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "assign_dialogue_speakers.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "assign_dialogue_speakers", SCRIPT_PATH
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
assign = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = assign
SCRIPT_SPEC.loader.exec_module(assign)

ACTIVE_PATH = ROOT / "data" / "speaker_pool" / "vctk_v0.1" / "active_speakers.json"
REGISTRY_PATH = ROOT / "data" / "speaker_pool" / "vctk_v0.1" / "speaker_registry.json"
SELECTION_PATH = ROOT / "data" / "speaker_pool" / "vctk_v0.1" / "selected_speakers.json"

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
EXCLUDED = ("spk_002", "spk_005")
SHORTLIST = tuple(f"spk_{number:03d}" for number in range(1, 13))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def _registry(
    path: Path,
    speaker_ids: tuple[str, ...] = SHORTLIST,
    *,
    version: str = "vctk_v0.1",
) -> Path:
    return _write_json(
        path,
        {
            "speaker_pool_version": version,
            "speakers": [{"speaker_id": speaker_id} for speaker_id in speaker_ids],
        },
    )


def _pool(
    path: Path,
    active: tuple[str, ...],
    excluded: tuple[tuple[str, str], ...] = (),
    *,
    version: str = "vctk_v0.1",
) -> Path:
    return _write_json(
        path,
        {
            "speaker_pool_version": version,
            "status": "active_pool_v0.1",
            "selection_basis": "test",
            "active_speaker_ids": list(active),
            "excluded_from_active_pool": [
                {"speaker_id": speaker_id, "reason": reason}
                for speaker_id, reason in excluded
            ],
        },
    )


def _turn(
    speaker: str,
    text: str = "Hello.",
    *,
    label: str = "normal",
    rate: str = "normal",
    arousal: str | None = None,
    coarse_affect: str | None = None,
    schema_v02: bool = True,
) -> dict[str, Any]:
    if schema_v02:
        return {
            "turn_id": 1,
            "speaker": speaker,
            "text": text,
            "label": label,
            "acoustic_spec": {
                "rate": rate,
                "pause_before_ms": 0,
                "pause_after_ms": 0,
                "arousal": arousal,
                "coarse_affect": coarse_affect,
                "emotion": None,
                "paralinguistic_events": [],
            },
        }
    payload: dict[str, Any] = {
        "turn_id": 1,
        "speaker": speaker,
        "text": text,
        "label": label,
        "rate": rate,
    }
    if arousal is not None:
        payload["arousal"] = arousal
    return payload


def _dialogue(
    path: Path,
    dialogue_id: str,
    role_turns: list[dict[str, Any]],
    *,
    schema_v02: bool = True,
) -> Path:
    turns = []
    for index, spec in enumerate(role_turns, start=1):
        turn = dict(spec)
        turn["turn_id"] = index
        turns.append(turn)
    payload: dict[str, Any] = {"dialogue_id": dialogue_id, "turns": turns}
    if schema_v02:
        payload["schema_version"] = "0.2"
    return _write_json(path, payload)


def _two_role_dialogue(
    directory: Path,
    dialogue_id: str,
    *,
    caller_label: str = "normal",
    counsellor_label: str = "normal",
    caller_rate: str = "normal",
    counsellor_rate: str = "normal",
    caller_arousal: str | None = None,
    counsellor_arousal: str | None = "medium",
    caller_affect: str | None = "distressed",
    counsellor_affect: str | None = None,
    schema_v02: bool = True,
    filename: str | None = None,
) -> Path:
    return _dialogue(
        directory / (filename or f"{dialogue_id}.json"),
        dialogue_id,
        [
            _turn(
                "caller",
                "I need to talk.",
                label=caller_label,
                rate=caller_rate,
                arousal=caller_arousal,
                coarse_affect=caller_affect,
                schema_v02=schema_v02,
            ),
            _turn(
                "counsellor",
                "I am here.",
                label=counsellor_label,
                rate=counsellor_rate,
                arousal=counsellor_arousal,
                coarse_affect=counsellor_affect,
                schema_v02=schema_v02,
            ),
        ],
        schema_v02=schema_v02,
    )


def _run(
    tmp_path: Path,
    dialogues_dir: Path,
    *,
    active: tuple[str, ...] = ("spk_001", "spk_003", "spk_004", "spk_006"),
    excluded: tuple[tuple[str, str], ...] = (
        ("spk_002", "pause behaviour"),
        ("spk_005", "pause behaviour"),
    ),
    registry_ids: tuple[str, ...] = SHORTLIST,
    version: str = "test_v0",
    seed: int = 5703,
    output_name: str = "speaker_assignments.jsonl",
) -> tuple[list[dict[str, Any]], dict[str, Any], Path, Path]:
    registry = _registry(
        tmp_path / "speaker_registry.json", registry_ids, version=version
    )
    pool = _pool(tmp_path / "active_speakers.json", active, excluded, version=version)
    output = tmp_path / output_name
    summary = tmp_path / "speaker_assignment_summary.json"
    assignments, summary_obj = assign.run_assignment(
        input_path=dialogues_dir,
        active_speakers_path=pool,
        registry_path=registry,
        output_path=output,
        summary_path=summary,
        seed=seed,
    )
    return assignments, summary_obj, output, summary


def test_active_pool_has_exactly_the_intended_ten_ids() -> None:
    pool = assign.load_and_validate_active_pool(ACTIVE_PATH, REGISTRY_PATH)
    assert pool.active_speaker_ids == EXPECTED_ACTIVE
    assert pool.speaker_pool_version == "vctk_v0.1"
    assert pool.status == "active_pool_v0.1"
    assert len(pool.active_speaker_ids) == 10


def test_spk_002_and_spk_005_are_excluded_from_the_active_pool() -> None:
    pool = assign.load_and_validate_active_pool(ACTIVE_PATH, REGISTRY_PATH)
    excluded = {item.speaker_id: item.reason for item in pool.exclusions}
    assert set(excluded) == set(EXCLUDED)
    assert "pause" in excluded["spk_002"].lower()
    assert "leading-silence" in excluded["spk_002"].lower()
    assert "pause" in excluded["spk_005"].lower()
    assert "spk_002" not in pool.active_speaker_ids
    assert "spk_005" not in pool.active_speaker_ids


def test_rejected_ids_remain_resolvable_in_registry_and_shortlist() -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    registry_ids = [entry["speaker_id"] for entry in registry["speakers"]]
    selection_ids = [entry["speaker_id"] for entry in selection["speakers"]]
    assert registry_ids == list(SHORTLIST)
    assert selection_ids == list(SHORTLIST)
    for speaker_id in EXCLUDED:
        assert speaker_id in registry_ids
        assert speaker_id in selection_ids


def test_speaker_ids_were_not_renumbered() -> None:
    pool = assign.load_and_validate_active_pool(ACTIVE_PATH, REGISTRY_PATH)
    assert "spk_013" not in pool.active_speaker_ids
    assert pool.active_speaker_ids[1] == "spk_003"
    assert list(pool.active_speaker_ids) != [f"spk_{n:03d}" for n in range(1, 11)]


def test_active_excluded_overlap_is_rejected(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json")
    pool = _pool(
        tmp_path / "active.json",
        EXPECTED_ACTIVE,
        (("spk_001", "should not also be active"),),
    )
    with pytest.raises(assign.SpeakerAssignmentError, match="both active and excluded"):
        assign.load_and_validate_active_pool(pool, registry)


def test_missing_active_registry_id_is_rejected(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json")
    pool = _pool(tmp_path / "active.json", EXPECTED_ACTIVE + ("spk_999",))
    with pytest.raises(assign.SpeakerAssignmentError, match="spk_999"):
        assign.load_and_validate_active_pool(pool, registry)


def test_pool_version_mismatch_is_rejected(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json", version="vctk_v0.1")
    pool = _pool(tmp_path / "active.json", EXPECTED_ACTIVE, version="vctk_v9.9")
    with pytest.raises(assign.SpeakerAssignmentError, match="vctk_v9.9"):
        assign.load_and_validate_active_pool(pool, registry)


def test_empty_exclusion_reason_is_rejected(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json")
    pool = _pool(tmp_path / "active.json", EXPECTED_ACTIVE, (("spk_002", "   "),))
    with pytest.raises(assign.SpeakerAssignmentError, match="reason"):
        assign.load_and_validate_active_pool(pool, registry)


def test_duplicate_active_ids_are_rejected(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json")
    pool = _pool(tmp_path / "active.json", ("spk_001", "spk_003", "spk_001"))
    with pytest.raises(assign.SpeakerAssignmentError, match="Duplicate active"):
        assign.load_and_validate_active_pool(pool, registry)


def test_v0_1_wrong_active_count_is_rejected(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json")
    pool = _pool(tmp_path / "active.json", EXPECTED_ACTIVE[:-1])
    with pytest.raises(assign.SpeakerAssignmentError, match="exactly 10"):
        assign.load_and_validate_active_pool(pool, registry)


def test_same_input_same_seed_is_byte_identical(tmp_path: Path) -> None:
    dialogues = tmp_path / "in"
    for index in range(6):
        _two_role_dialogue(dialogues, f"D{index:03d}")
    first = _run(tmp_path / "a", dialogues)
    second = _run(tmp_path / "b", dialogues)
    assert first[2].read_bytes() == second[2].read_bytes()
    assert first[3].read_bytes() == second[3].read_bytes()
    assert first[0] == second[0]
    assert first[1] == second[1]


def test_different_seed_can_change_a_valid_assignment(tmp_path: Path) -> None:
    dialogues = tmp_path / "in"
    for index in range(8):
        _two_role_dialogue(dialogues, f"D{index:03d}")
    seed_a = _run(tmp_path / "a", dialogues, seed=5703)
    seed_b = _run(tmp_path / "b", dialogues, seed=5704)
    assert seed_a[0] != seed_b[0]
    for record in seed_b[0]:
        assert (
            record["role_assignments"]["caller"]
            != record["role_assignments"]["counsellor"]
        )


def test_input_filesystem_order_does_not_change_output(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    specs = (("D002", "z.json"), ("D001", "a.json"), ("D003", "m.json"))
    for dialogue_id, name in specs:
        _two_role_dialogue(first_dir, dialogue_id, filename=name)
    for dialogue_id, name in reversed(specs):
        _two_role_dialogue(second_dir, dialogue_id, filename=f"other_{name}")
    first = _run(tmp_path / "out_a", first_dir)
    second = _run(tmp_path / "out_b", second_dir)
    assert [record["dialogue_id"] for record in first[0]] == ["D001", "D002", "D003"]
    assert first[2].read_bytes() == second[2].read_bytes()
    assert first[3].read_bytes() == second[3].read_bytes()


def test_caller_and_counsellor_receive_distinct_speakers(tmp_path: Path) -> None:
    dialogues = tmp_path / "in"
    for index in range(12):
        _two_role_dialogue(dialogues, f"D{index:03d}")
    assignments, _, _, _ = _run(tmp_path / "out", dialogues)
    for record in assignments:
        mapping = record["role_assignments"]
        assert mapping["caller"] != mapping["counsellor"]


def test_arbitrary_role_names_are_supported(tmp_path: Path) -> None:
    dialogues = tmp_path / "in"
    _dialogue(
        dialogues / "scene.json",
        "scene_01",
        [
            _turn("client", "I am tired."),
            _turn("therapist", "Tell me more."),
            _turn("supervisor", "We can pause."),
        ],
    )
    assignments, _, _, _ = _run(tmp_path / "out", dialogues)
    mapping = assignments[0]["role_assignments"]
    assert set(mapping) == {"client", "therapist", "supervisor"}
    assert len(set(mapping.values())) == 3


def test_n_role_dialogue_assigns_n_distinct_speakers(tmp_path: Path) -> None:
    dialogues = tmp_path / "in"
    roles = ("alpha", "beta", "gamma", "delta")
    _dialogue(
        dialogues / "many.json",
        "many_01",
        [_turn(role, f"Turn {role}.") for role in roles],
    )
    assignments, _, _, _ = _run(tmp_path / "out", dialogues)
    mapping = assignments[0]["role_assignments"]
    assert list(mapping) == list(roles)
    assert len(set(mapping.values())) == len(roles)


def test_too_many_roles_fails_clearly(tmp_path: Path) -> None:
    dialogues = tmp_path / "in"
    roles = tuple(f"role_{index}" for index in range(5))
    _dialogue(
        dialogues / "overflow.json",
        "overflow_01",
        [_turn(role, f"Turn {role}.") for role in roles],
    )
    with pytest.raises(assign.SpeakerAssignmentError, match="exceed the 4"):
        _run(
            tmp_path / "out",
            dialogues,
            active=("spk_001", "spk_003", "spk_004", "spk_006"),
        )


def test_assigned_ids_are_active_and_never_rejected(tmp_path: Path) -> None:
    dialogues = tmp_path / "in"
    active = ("spk_001", "spk_003", "spk_004", "spk_006")
    for index in range(10):
        _two_role_dialogue(dialogues, f"D{index:03d}")
    assignments, _, output, _ = _run(tmp_path / "out", dialogues, active=active)
    assigned = {
        speaker_id
        for record in assignments
        for speaker_id in record["role_assignments"].values()
    }
    assert assigned <= set(active)
    assert assigned.isdisjoint(EXCLUDED)
    written = output.read_text(encoding="utf-8")
    assert "spk_002" not in written
    assert "spk_005" not in written
    assert "prompt_wav" not in written
    assert "prompt_text" not in written


def test_speaker_ids_recur_and_usage_is_reasonably_balanced(tmp_path: Path) -> None:
    dialogues = tmp_path / "in"
    active = ("spk_001", "spk_003", "spk_004", "spk_006")
    for index in range(10):
        _two_role_dialogue(dialogues, f"D{index:03d}")
    assignments, summary, _, _ = _run(tmp_path / "out", dialogues, active=active)
    usage = summary["speaker_usage"]
    assert max(usage.values()) - min(usage.values()) <= 1
    recurring = [speaker_id for speaker_id, count in usage.items() if count >= 2]
    assert len(recurring) == len(active)
    appeared = {
        speaker_id
        for record in assignments
        for speaker_id in record["role_assignments"].values()
    }
    assert appeared == set(active)


def test_role_usage_is_reasonably_distributed(tmp_path: Path) -> None:
    dialogues = tmp_path / "in"
    active = ("spk_001", "spk_003", "spk_004", "spk_006")
    for index in range(12):
        _two_role_dialogue(dialogues, f"D{index:03d}")
    _, summary, _, _ = _run(tmp_path / "out", dialogues, active=active)
    for role in ("caller", "counsellor"):
        counts = list(summary["role_usage"][role].values())
        assert max(counts) - min(counts) <= 2
    both_roles = [
        speaker_id
        for speaker_id in active
        if summary["role_usage"]["caller"][speaker_id]
        and summary["role_usage"]["counsellor"][speaker_id]
    ]
    assert both_roles, "speakers should not be permanently tied to one role"


def test_labels_do_not_participate_in_assignment(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    for index in range(5):
        _two_role_dialogue(
            left,
            f"D{index:03d}",
            caller_label="alert",
            counsellor_label="confirm",
            caller_arousal="high",
            caller_affect="distressed",
        )
        _two_role_dialogue(
            right,
            f"D{index:03d}",
            caller_label="normal",
            counsellor_label="normal",
            caller_arousal=None,
            caller_affect=None,
            counsellor_rate="fast",
        )
    left_run = _run(tmp_path / "out_left", left)
    right_run = _run(tmp_path / "out_right", right)
    left_roles = [record["role_assignments"] for record in left_run[0]]
    right_roles = [record["role_assignments"] for record in right_run[0]]
    assert left_roles == right_roles
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    function = source.split("def choose_speaker_for_role", 1)[1].split("\ndef ", 1)[0]
    assert "label" not in function.split('"""', 2)[2]
    assert "arousal" not in function.split('"""', 2)[2]
    assert "coarse_affect" not in function.split('"""', 2)[2]
    assert "emotion" not in function.split('"""', 2)[2]


def test_label_and_acoustic_exposure_match_assigned_turns(tmp_path: Path) -> None:
    dialogues = tmp_path / "in"
    _two_role_dialogue(
        dialogues,
        "D001",
        caller_label="alert",
        counsellor_label="confirm",
        caller_rate="slow",
        counsellor_rate="fast",
        caller_arousal="high",
        counsellor_arousal="low",
        caller_affect="distressed",
        counsellor_affect=None,
        filename="keep.json",
    )
    _two_role_dialogue(
        dialogues,
        "D002",
        caller_label="normal",
        counsellor_label="normal",
        caller_rate="normal",
        counsellor_rate="normal",
        caller_arousal=None,
        counsellor_arousal="medium",
        caller_affect=None,
        counsellor_affect=None,
    )
    assignments, summary, _, _ = _run(tmp_path / "out", dialogues)
    expected_labels: dict[str, dict[str, int]] = {
        speaker_id: {key: 0 for key in ("normal", "alert", "confirm")}
        for speaker_id in ("spk_001", "spk_003", "spk_004", "spk_006")
    }
    expected_rate: dict[str, dict[str, int]] = {
        speaker_id: {} for speaker_id in expected_labels
    }
    expected_arousal: dict[str, dict[str, int]] = {
        speaker_id: {} for speaker_id in expected_labels
    }
    expected_affect: dict[str, dict[str, int]] = {
        speaker_id: {} for speaker_id in expected_labels
    }
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in dialogues.glob("*.json")
    ]
    mapping_by_id = {
        record["dialogue_id"]: record["role_assignments"] for record in assignments
    }
    for payload in records:
        mapping = mapping_by_id[payload["dialogue_id"]]
        for turn in payload["turns"]:
            speaker_id = mapping[turn["speaker"]]
            spec = turn["acoustic_spec"]
            expected_labels[speaker_id][turn["label"]] += 1
            expected_rate[speaker_id][spec["rate"]] = (
                expected_rate[speaker_id].get(spec["rate"], 0) + 1
            )
            arousal_key = spec["arousal"] if spec["arousal"] is not None else "null"
            affect_key = (
                spec["coarse_affect"] if spec["coarse_affect"] is not None else "null"
            )
            expected_arousal[speaker_id][arousal_key] = (
                expected_arousal[speaker_id].get(arousal_key, 0) + 1
            )
            expected_affect[speaker_id][affect_key] = (
                expected_affect[speaker_id].get(affect_key, 0) + 1
            )
    for speaker_id, counts in expected_labels.items():
        for label, count in counts.items():
            assert summary["label_exposure"][speaker_id][label] == count
    for speaker_id, counts in expected_rate.items():
        for rate, count in counts.items():
            assert summary["acoustic_exposure"][speaker_id]["rate"][rate] == count
    for speaker_id, counts in expected_arousal.items():
        for key, count in counts.items():
            assert summary["acoustic_exposure"][speaker_id]["arousal"][key] == count
    for speaker_id, counts in expected_affect.items():
        for key, count in counts.items():
            assert (
                summary["acoustic_exposure"][speaker_id]["coarse_affect"][key] == count
            )


def test_assignment_does_not_use_builtin_hash() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "hashlib.sha256" in source
    assert re.search(r"(?<![\w.])hash\(", source) is None
    assert "random.choice" not in source


def test_jsonl_and_summary_are_deterministic_across_reruns(tmp_path: Path) -> None:
    dialogues = tmp_path / "in"
    for index in range(4):
        _two_role_dialogue(dialogues, f"D{index:03d}")
    first = _run(tmp_path / "one", dialogues, seed=5703)
    second = _run(tmp_path / "two", dialogues, seed=5703)
    assert first[2].read_bytes() == second[2].read_bytes()
    assert first[3].read_bytes() == second[3].read_bytes()
    assert first[2].read_text(encoding="utf-8").endswith("\n")
    assert first[3].read_text(encoding="utf-8").endswith("\n")


def test_cli_writes_manifests(tmp_path: Path) -> None:
    dialogues = tmp_path / "in"
    _two_role_dialogue(dialogues, "D001")
    registry = _registry(tmp_path / "registry.json", version="test_v0")
    pool = _pool(
        tmp_path / "active.json",
        ("spk_001", "spk_003", "spk_004", "spk_006"),
        version="test_v0",
    )
    output = tmp_path / "out" / "speaker_assignments.jsonl"
    summary = tmp_path / "out" / "speaker_assignment_summary.json"
    exit_code = assign.main(
        [
            "--input",
            str(dialogues),
            "--active-speakers",
            str(pool),
            "--speaker-registry",
            str(registry),
            "--output",
            str(output),
            "--summary",
            str(summary),
            "--seed",
            "5703",
        ]
    )
    assert exit_code == 0
    assert output.is_file()
    record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert record["dialogue_id"] == "D001"
    assert record["assignment_seed"] == 5703
    assert set(record["role_assignments"]) == {"caller", "counsellor"}
