"""Offline contracts for the frozen v0.2 production speaker mapping."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import assign_production_speakers as production


@pytest.fixture(scope="module")
def canonical():
    dialogues = production.input_helpers.load_dialogues(
        production.ROOT / production.INPUT
    )
    voices = production.load_voices()
    exposure = production.collect_exposure(dialogues)
    policy, rows, summary = production.build_artifacts()
    return dialogues, voices, exposure, policy, rows, summary


def test_canonical_bytes_and_historical_files_are_unchanged(
    canonical, tmp_path: Path
) -> None:
    _, _, _, policy, rows, summary = canonical
    second_policy, second_rows, second_summary = production.build_artifacts()
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    production.write_artifacts(policy, rows, summary, first_dir)
    production.write_artifacts(second_policy, second_rows, second_summary, second_dir)
    for name in ("policy.json", "speaker_assignments.jsonl", "summary.json"):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
    assert len(policy["identity_map"]) == 60
    assert len(rows) == 1000
    assert summary["corpus"]["turn_count"] == 15934
    assert production.sha256(production.ROOT / production.INPUT) == (
        "137d74faf441a8e040fe8534b1e17439df541b51f2a3ae2b7e2bc64390027fb0"
    )
    assert (
        production.sha256(production.ROOT / "data/speaker_assignments.jsonl")
        == "a02281819e3a083352f925f9413a7588821badc57daaf12f6a680abf5a48329a"
    )
    assert (
        production.sha256(
            production.ROOT / "data/speaker_pool/vctk_v0.1/speaker_assignments.jsonl"
        )
        == "714d38715d78bc3965dbddec1fdd721b1ed39c9aa3340320beb446a58918003e"
    )


def test_record_order_and_age_band_do_not_choose_voices(canonical) -> None:
    dialogues, voices, exposure, policy, _, _ = canonical
    reversed_exposure = production.collect_exposure(list(reversed(dialogues)))
    assert (
        production.assign_identity_map(reversed_exposure, voices)
        == policy["identity_map"]
    )

    changed_age = deepcopy(exposure)
    changed_age["C000"]["requested_age_band"] = "different_audit_label"
    remapped = production.assign_identity_map(changed_age, voices)
    assert {
        upstream: item["render_speaker_id"] for upstream, item in remapped.items()
    } == {
        upstream: item["render_speaker_id"]
        for upstream, item in policy["identity_map"].items()
    }


def test_frozen_map_controls_dialogue_subsets(canonical, monkeypatch) -> None:
    dialogues, _, _, policy, rows, _ = canonical

    def refuse_recompute(*_args, **_kwargs):
        raise AssertionError("subset rendering must not recompute the identity map")

    monkeypatch.setattr(production, "assign_identity_map", refuse_recompute)
    subset = dialogues[100:115]
    subset_ids = {dialogue.dialogue_id for dialogue in subset}
    assert production.assignment_rows(subset, policy["identity_map"]) == [
        row for row in rows if row["dialogue_id"] in subset_ids
    ]


def test_gender_role_partition_and_turn_balancing(canonical) -> None:
    _, voices, _, policy, rows, summary = canonical
    assert policy["fixed_role_partition"] == {
        "caller": {
            "F": ("spk_001", "spk_003", "spk_004", "spk_014", "spk_015"),
            "M": ("spk_008", "spk_009", "spk_010", "spk_011"),
        },
        "counsellor": {
            "F": ("spk_006", "spk_016", "spk_017"),
            "M": ("spk_007", "spk_012", "spk_013"),
        },
    }
    assert len(policy["fixed_role_partition"]["caller"]["F"]) == 5
    assert len(policy["fixed_role_partition"]["caller"]["M"]) == 4
    assert len(policy["fixed_role_partition"]["counsellor"]["F"]) == 3
    assert len(policy["fixed_role_partition"]["counsellor"]["M"]) == 3
    partition = summary["role_partition"]
    assert partition["intersection"] == []
    assert len(partition["caller_voices"]) == 9
    assert len(partition["counsellor_voices"]) == 6
    assert set(partition["caller_voices"] + partition["counsellor_voices"]) == set(
        voices
    )
    assert summary["mapping"]["all_15_voices_used"] is True
    assert all(
        stats["identity_count"] > 0
        for stats in summary["mapping"]["per_render_voice"].values()
    )

    for upstream, item in policy["identity_map"].items():
        speaker_id = item["render_speaker_id"]
        assert voices[speaker_id]["role"] == item["role"]
        if item["requested_gender"] == "female":
            assert item["render_source_gender"] == "F"
        elif item["requested_gender"] == "male":
            assert item["render_source_gender"] == "M"
        else:
            assert item["requested_gender"] == "neutral"
            assert item["render_source_gender"] in {"F", "M"}
        assert upstream.startswith("C") == (item["role"] == "caller")
    assert all(
        row["role_assignments"]["caller"] != row["role_assignments"]["counsellor"]
        for row in rows
    )
    assert summary["gender"]["neutral_to_source"]["F"]["identity_count"] == 10
    assert summary["gender"]["neutral_to_source"]["M"]["identity_count"] == 10
    assert summary["gender"]["neutral_to_source"]["F"]["turn_count"] == 2806
    assert summary["gender"]["neutral_to_source"]["M"]["turn_count"] == 2748


def test_unstable_upstream_metadata_fails_closed(canonical) -> None:
    dialogues, _, _, _, _, _ = canonical
    first = dialogues[0]
    upstream = first.raw["scenario"]["speakers"]["caller"]["speaker_id"]
    second = next(
        dialogue
        for dialogue in dialogues[1:]
        if dialogue.raw["scenario"]["speakers"]["caller"]["speaker_id"] == upstream
    )
    changed = deepcopy(second.raw)
    original_gender = changed["scenario"]["speakers"]["caller"]["gender"]
    changed["scenario"]["speakers"]["caller"]["gender"] = (
        "male" if original_gender != "male" else "female"
    )
    with pytest.raises(ValueError, match="Unstable role/gender/age_band"):
        production.collect_exposure([first, replace(second, raw=changed)])


def test_review_only_higgs_reference_fails_pool_load(tmp_path: Path) -> None:
    registry = json.loads((production.ROOT / production.REGISTRY).read_text())
    registry["speakers"][0]["higgs_reference"]["approval_status"] = "review_only"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="not production approved"):
        production.load_voices(path, production.ROOT / production.ACTIVE)
