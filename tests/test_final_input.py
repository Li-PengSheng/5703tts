from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from tts5703.final_input import (
    FinalInputValidationError,
    SchemaFamily,
    SchemaFamilyError,
    detect_schema_family,
    validate_final_dialogue,
)


def _final_dialogue() -> dict[str, Any]:
    required = {
        "rate": "normal",
        "arousal": 2,
        "affect": "neutral",
        "pause_before": "short",
        "pause_within": 2,
        "hesitations": 1,
    }
    return {
        "schema_version": "1.0",
        "dialogue_id": "D-final",
        "scenario": {
            "speakers": {
                "caller": {"speaker_id": "C123"},
                "counsellor": {"speaker_id": "L456"},
            }
        },
        "turns": [
            {
                "turn_id": 1,
                "speaker": "User",
                "text": "I need help.",
                "labels": {"topics": ["synthetic", "opaque"]},
                "acoustic": {
                    "required": required,
                    "best_effort": {
                        "affect_fine": "worried",
                        "volume": "soft",
                        "flattened_affect": False,
                        "events": ["door"],
                    },
                },
            },
            {
                "turn_id": 2,
                "speaker": "Listener",
                "text": "I am listening.",
                "labels": ["also", "opaque"],
                "acoustic": {"required": {**required, "pause_before": "none"}},
            },
        ],
    }


def test_schema_family_detection_is_structural() -> None:
    assert (
        detect_schema_family({"dialogue_id": "L", "turns": []}) is SchemaFamily.LEGACY
    )
    assert (
        detect_schema_family(
            {
                "schema_version": "0.2",
                "dialogue_id": "V",
                "turns": [{"acoustic_spec": {}}],
            }
        )
        is SchemaFamily.V0_2
    )
    assert detect_schema_family(_final_dialogue()) is SchemaFamily.FINAL_NESTED


def test_final_schema_version_is_not_routed_to_v02() -> None:
    final = _final_dialogue()
    final["schema_version"] = "future-final"
    assert detect_schema_family(final) is SchemaFamily.FINAL_NESTED


@pytest.mark.parametrize(
    "turns",
    [
        None,
        [],
        {},
        [None],
        [{"speaker": "caller", "rate": "normal"}],
        [{"acoustic_spec": []}],
    ],
)
def test_v02_requires_nonempty_well_formed_turns(turns: Any) -> None:
    raw = {"schema_version": "0.2", "dialogue_id": "not-v02"}
    if turns is not None:
        raw["turns"] = turns

    with pytest.raises(SchemaFamilyError, match="requires"):
        detect_schema_family(raw)


@pytest.mark.parametrize(
    "hybrid",
    [
        {
            "schema_version": "0.2",
            "turns": [{"acoustic": {"required": {}}}],
        },
        {
            "turns": [
                {"acoustic_spec": {}},
                {"acoustic": {"required": {}}},
            ]
        },
        {
            "turns": [
                {"acoustic_spec": {}, "acoustic": {"required": {}}},
            ]
        },
    ],
)
def test_hybrid_schema_is_rejected(hybrid: dict[str, Any]) -> None:
    with pytest.raises(SchemaFamilyError, match="Hybrid"):
        detect_schema_family(hybrid)


def test_final_validation_preserves_nested_source_and_speaker_identities() -> None:
    source = _final_dialogue()
    before = deepcopy(source)

    validated = validate_final_dialogue(source)

    assert source == before
    assert validated.raw == before
    assert validated.raw["turns"][0]["labels"] == {"topics": ["synthetic", "opaque"]}
    assert validated.raw["turns"][0]["acoustic"]["required"]["arousal"] == 2
    assert validated.speaker_identities["caller"].upstream_role == "User"
    assert validated.speaker_identities["caller"].upstream_scenario_speaker_id == "C123"
    assert (
        validated.speaker_identities["counsellor"].upstream_scenario_speaker_id
        == "L456"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rate", "very_fast"),
        ("arousal", True),
        ("arousal", 4),
        ("affect", "joyful"),
        ("pause_before", "medium"),
        ("pause_within", -1),
        ("hesitations", 3),
    ],
)
def test_invalid_final_required_control_fails_before_rendering(
    field: str, value: Any
) -> None:
    final = _final_dialogue()
    final["turns"][0]["acoustic"]["required"][field] = value

    with pytest.raises(FinalInputValidationError, match=field):
        validate_final_dialogue(final)
