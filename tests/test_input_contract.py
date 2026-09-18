from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from tts5703.input_contract import (
    FinalInputValidationError,
    validate_dialogue,
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


@pytest.mark.parametrize(
    "raw",
    [
        {"dialogue_id": "legacy", "turns": []},
        {
            "schema_version": "0.2",
            "dialogue_id": "old",
            "turns": [{"turn_id": 1, "speaker": "caller", "acoustic_spec": {}}],
        },
    ],
)
def test_non_final_contracts_fail_validation(raw: dict[str, Any]) -> None:
    with pytest.raises(FinalInputValidationError):
        validate_dialogue(raw)


def test_final_validation_preserves_nested_source_and_speaker_identities() -> None:
    source = _final_dialogue()
    before = deepcopy(source)

    validated = validate_dialogue(source)

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
        validate_dialogue(final)
