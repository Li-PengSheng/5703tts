from __future__ import annotations

import json
from pathlib import Path

import pytest

from tts5703.exclusion_policy import (
    ExclusionPolicyError,
    load_exclusion_policy,
    parse_exclusion_policy,
)


def _policy() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "exclusions": [
            {
                "dialogue_id": "synthetic_002",
                "reason": "Synthetic known issue",
                "provenance": {
                    "source": "unit-test",
                    "details": {"tickets": [2]},
                },
            },
            {
                "dialogue_id": "synthetic_001",
                "reason": "Another synthetic issue",
            },
        ],
    }


def test_valid_policy_loads_and_lookup_works(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(_policy()), encoding="utf-8")

    policy = load_exclusion_policy(path)

    assert policy.schema_version == "1.0"
    assert policy.exclusion_for("synthetic_002").reason == "Synthetic known issue"
    assert policy.exclusion_for("not-listed") is None


def test_provenance_is_isolated_from_source_and_lookup_mutations() -> None:
    source_policy = _policy()
    policy = parse_exclusion_policy(source_policy)
    expected_hash = policy.policy_sha256

    source_policy["exclusions"][0]["provenance"]["details"]["tickets"].append(3)
    first = policy.exclusion_for("synthetic_002")
    assert first is not None
    assert first.provenance["details"]["tickets"] == [2]

    first.provenance["details"]["tickets"].append(4)
    second = policy.exclusion_for("synthetic_002")
    assert second is not None
    assert second.provenance["details"]["tickets"] == [2]
    assert policy.exclusions[1].provenance["details"]["tickets"] == [2]
    assert policy.policy_sha256 == expected_hash


def test_duplicate_dialogue_ids_are_rejected() -> None:
    raw = _policy()
    raw["exclusions"].append({"dialogue_id": "synthetic_001", "reason": "duplicate"})
    with pytest.raises(ExclusionPolicyError, match="Duplicate dialogue_id"):
        parse_exclusion_policy(raw)


@pytest.mark.parametrize(
    "entry",
    [
        {"dialogue_id": " ", "reason": "reason"},
        {"dialogue_id": "synthetic", "reason": ""},
    ],
)
def test_blank_ids_and_reasons_are_rejected(entry: dict[str, str]) -> None:
    with pytest.raises(ExclusionPolicyError, match="nonblank"):
        parse_exclusion_policy({"schema_version": "1.0", "exclusions": [entry]})


def test_policy_hash_is_semantic_not_format_or_key_order_dependent(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    raw = _policy()
    left.write_text(json.dumps(raw, indent=4), encoding="utf-8")
    right.write_text(
        json.dumps(
            {
                "exclusions": [
                    {key: value for key, value in reversed(list(item.items()))}
                    for item in reversed(raw["exclusions"])
                ],
                "schema_version": "1.0",
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    assert (
        load_exclusion_policy(left).policy_sha256
        == load_exclusion_policy(right).policy_sha256
    )


def test_no_production_corpus_id_is_embedded_in_policy_module() -> None:
    source = Path("src/tts5703/exclusion_policy.py").read_text(encoding="utf-8")
    assert "corpus_v1_" not in source


def test_policy_json_rejects_duplicate_object_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version":"1.0","exclusions":[],"exclusions":[]}',
        encoding="utf-8",
    )

    with pytest.raises(ExclusionPolicyError, match="duplicate object key"):
        load_exclusion_policy(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_policy_json_rejects_non_json_numeric_constants(
    tmp_path: Path, constant: str
) -> None:
    path = tmp_path / "constant.json"
    path.write_text(
        '{"schema_version":"1.0","exclusions":[],"extra":' + constant + "}",
        encoding="utf-8",
    )

    with pytest.raises(ExclusionPolicyError, match="invalid JSON constant"):
        load_exclusion_policy(path)
