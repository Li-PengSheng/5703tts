"""Offline structural and experimental-control checks for benchmark v0.1."""

import json
from pathlib import Path
from typing import Any

import pytest

from tts5703.config import load_config
from tts5703.validate import load_and_validate

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data" / "benchmark"
FIXTURE_PATH = BENCHMARK_DIR / "benchmark_fixture_v0.1.json"
MANIFEST_PATH = BENCHMARK_DIR / "benchmark_manifest_v0.1.json"
CONFIG_PATH = ROOT / "config" / "config.yaml"
TARGET_VARIABLES = {
    "rate": "rate",
    "pause_before": "pause_before_ms",
    "pause_after": "pause_after_ms",
    "arousal": "arousal",
    "coarse_affect": "coarse_affect",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _group_turns(
    fixture: dict[str, Any], manifest: dict[str, Any], group_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    group = next(group for group in manifest["groups"] if group["group_id"] == group_id)
    turns_by_id = {turn["turn_id"]: turn for turn in fixture["turns"]}
    return group, [turns_by_id[turn_id] for turn_id in group["turn_ids"]]


def test_benchmark_files_exist() -> None:
    assert FIXTURE_PATH.is_file()
    assert MANIFEST_PATH.is_file()
    assert (BENCHMARK_DIR / "README.md").is_file()


def test_fixture_validates_through_canonical_pipeline_path() -> None:
    dialogue = load_and_validate(FIXTURE_PATH, load_config(CONFIG_PATH))
    assert dialogue.dialogue_id == "controlled_tts_benchmark_v0_1"
    assert len(dialogue.turns) == 11


def test_fixture_is_canonical_v0_2_not_legacy_flat_input() -> None:
    fixture = _load_json(FIXTURE_PATH)
    acoustic_fields = {
        "rate",
        "pause_before_ms",
        "pause_after_ms",
        "arousal",
        "coarse_affect",
        "emotion",
        "paralinguistic_events",
    }

    assert fixture["schema_version"] == "0.2"
    assert all("acoustic_spec" in turn for turn in fixture["turns"])
    assert all(not acoustic_fields.intersection(turn) for turn in fixture["turns"])


def test_fixture_has_exactly_eleven_turns_and_both_roles() -> None:
    turns = _load_json(FIXTURE_PATH)["turns"]
    assert len(turns) == 11
    assert {turn["speaker"] for turn in turns} == {"caller", "counsellor"}


def test_spoken_text_lengths_are_within_design_range() -> None:
    turns = _load_json(FIXTURE_PATH)["turns"]
    assert all(8 <= len(turn["text"].split()) <= 20 for turn in turns)


def test_manifest_groups_reference_valid_turns() -> None:
    fixture = _load_json(FIXTURE_PATH)
    manifest = _load_json(MANIFEST_PATH)
    valid_ids = {turn["turn_id"] for turn in fixture["turns"]}
    referenced_ids = [
        turn_id for group in manifest["groups"] for turn_id in group["turn_ids"]
    ]

    assert manifest["fixture"] == FIXTURE_PATH.name
    assert set(referenced_ids).issubset(valid_ids)
    assert sorted(referenced_ids) == sorted(valid_ids)
    assert all(len(group["turn_ids"]) >= 2 for group in manifest["groups"])
    assert all(
        group["baseline_turn_id"] in group["turn_ids"] for group in manifest["groups"]
    )


def test_manifest_defines_the_five_hard_control_groups() -> None:
    manifest = _load_json(MANIFEST_PATH)
    assert {
        group["group_id"]: group["target_variable"] for group in manifest["groups"]
    } == TARGET_VARIABLES


@pytest.mark.parametrize(
    ("group_id", "expected_conditions"),
    [
        ("rate", ["slow", "normal", "fast"]),
        ("arousal", ["low", "high"]),
    ],
)
def test_required_semantic_conditions(
    group_id: str, expected_conditions: list[str]
) -> None:
    fixture = _load_json(FIXTURE_PATH)
    manifest = _load_json(MANIFEST_PATH)
    group, turns = _group_turns(fixture, manifest, group_id)

    assert group["expected_conditions"] == expected_conditions
    assert [
        turn["acoustic_spec"][group["target_variable"]] for turn in turns
    ] == expected_conditions


@pytest.mark.parametrize("group_id", list(TARGET_VARIABLES))
def test_group_members_share_text_speaker_and_label(group_id: str) -> None:
    fixture = _load_json(FIXTURE_PATH)
    manifest = _load_json(MANIFEST_PATH)
    _, turns = _group_turns(fixture, manifest, group_id)

    assert len({turn["text"] for turn in turns}) == 1
    assert len({turn["speaker"] for turn in turns}) == 1
    assert len({turn["label"] for turn in turns}) == 1


@pytest.mark.parametrize("group_id", list(TARGET_VARIABLES))
def test_each_group_changes_only_its_target_acoustic_variable(group_id: str) -> None:
    fixture = _load_json(FIXTURE_PATH)
    manifest = _load_json(MANIFEST_PATH)
    group, turns = _group_turns(fixture, manifest, group_id)
    target = group["target_variable"]
    non_target_specs = [
        {key: value for key, value in turn["acoustic_spec"].items() if key != target}
        for turn in turns
    ]
    actual_conditions = [turn["acoustic_spec"][target] for turn in turns]

    assert all(spec == non_target_specs[0] for spec in non_target_specs[1:])
    assert actual_conditions == group["expected_conditions"]
    assert len(set(actual_conditions)) == len(actual_conditions)


def test_best_effort_fields_do_not_confound_hard_control_groups() -> None:
    turns = _load_json(FIXTURE_PATH)["turns"]
    assert all(turn["acoustic_spec"]["emotion"] is None for turn in turns)
    assert all(turn["acoustic_spec"]["paralinguistic_events"] == [] for turn in turns)
