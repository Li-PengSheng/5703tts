"""Fast validation boundary checks; no TTS/network required."""

import copy
import importlib
import logging
from pathlib import Path

import pytest

config_module = importlib.import_module("tts5703.config")
validate_module = importlib.import_module("tts5703.validate")
load_config = config_module.load_config
ConfigError = config_module.ConfigError
validate_config = config_module._validate_config
ValidationError = validate_module.ValidationError
validate_and_normalize = validate_module.validate_and_normalize
load_and_validate = validate_module.load_and_validate

CONFIG = load_config(Path("config/config.yaml"))
LEGACY_MINIMAL = {
    "dialogue_id": "ok001",
    "turns": [
        {"turn_id": 1, "speaker": "counsellor", "text": "Hi.", "label": "normal"},
        {"turn_id": 2, "speaker": "caller", "text": "Hello.", "label": "normal"},
    ],
}


def _canonical_dialogue(acoustic_spec: dict | None = None) -> dict:
    return {
        "schema_version": "0.2",
        "dialogue_id": "canonical001",
        "turns": [
            {
                "turn_id": 1,
                "speaker": "caller",
                "text": "I need to talk.",
                "label": "alert",
                "acoustic_spec": acoustic_spec if acoustic_spec is not None else {},
            }
        ],
    }


def test_legacy_minimal_dialogue_still_works() -> None:
    dialogue = validate_and_normalize(LEGACY_MINIMAL, CONFIG)
    assert dialogue.dialogue_id == "ok001"
    assert dialogue.turns[0].rate == "+0%"
    assert dialogue.turns[0].pause_after_ms == 500


def test_legacy_input_emits_temporary_support_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="tts5703.validate"):
        validate_and_normalize(LEGACY_MINIMAL, CONFIG)

    assert any(
        "Legacy dialogue schema is temporarily supported" in record.message
        and "schema v0.2 is preferred" in record.message
        for record in caplog.records
    )


def test_canonical_input_does_not_emit_legacy_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="tts5703.validate"):
        validate_and_normalize(_canonical_dialogue(), CONFIG)

    assert not any(
        "Legacy dialogue schema is temporarily supported" in record.message
        for record in caplog.records
    )


def test_canonical_v0_2_fixture_works() -> None:
    dialogue = load_and_validate(Path("tests/fixtures/dialogue_v0_2.json"), CONFIG)
    assert dialogue.dialogue_id == "fixture_v0_2"
    assert [turn.speaker for turn in dialogue.turns] == ["caller", "counsellor"]


@pytest.mark.parametrize("rate", ["slow", "normal", "fast"])
def test_semantic_rates_are_accepted(rate: str) -> None:
    dialogue = validate_and_normalize(_canonical_dialogue({"rate": rate}), CONFIG)
    assert dialogue.turns[0].rate == rate


@pytest.mark.parametrize("arousal", ["low", "medium", "high"])
def test_arousal_levels_are_accepted(arousal: str) -> None:
    dialogue = validate_and_normalize(_canonical_dialogue({"arousal": arousal}), CONFIG)
    assert dialogue.turns[0].arousal == arousal


def test_canonical_optional_values_are_accepted() -> None:
    dialogue = validate_and_normalize(
        _canonical_dialogue(
            {
                "pause_before_ms": 0,
                "pause_after_ms": 700,
                "coarse_affect": "distressed",
                "emotion": None,
                "paralinguistic_events": [],
            }
        ),
        CONFIG,
    )
    turn = dialogue.turns[0]
    assert turn.pause_before_ms == 0
    assert turn.pause_after_ms == 700
    assert turn.coarse_affect == "distressed"
    assert turn.emotion is None
    assert turn.paralinguistic_events == []


def test_null_coarse_affect_is_accepted() -> None:
    dialogue = validate_and_normalize(
        _canonical_dialogue({"coarse_affect": None}), CONFIG
    )
    assert dialogue.turns[0].coarse_affect is None


def test_missing_dialogue_id_is_rejected() -> None:
    invalid = _canonical_dialogue()
    del invalid["dialogue_id"]
    with pytest.raises(ValidationError):
        validate_and_normalize(invalid, CONFIG)


def test_empty_dialogue_id_is_rejected() -> None:
    invalid = _canonical_dialogue()
    invalid["dialogue_id"] = ""
    with pytest.raises(ValidationError):
        validate_and_normalize(invalid, CONFIG)


def test_empty_text_is_rejected() -> None:
    invalid = _canonical_dialogue()
    invalid["turns"][0]["text"] = ""
    with pytest.raises(ValidationError):
        validate_and_normalize(invalid, CONFIG)


def test_invalid_label_is_rejected() -> None:
    invalid = _canonical_dialogue()
    invalid["turns"][0]["label"] = "urgent"
    with pytest.raises(ValidationError):
        validate_and_normalize(invalid, CONFIG)


def test_invalid_semantic_rate_is_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_and_normalize(_canonical_dialogue({"rate": "very_slow"}), CONFIG)


def test_invalid_arousal_is_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_and_normalize(_canonical_dialogue({"arousal": "extreme"}), CONFIG)


@pytest.mark.parametrize("field", ["pause_before_ms", "pause_after_ms"])
def test_negative_pauses_are_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        validate_and_normalize(_canonical_dialogue({field: -1}), CONFIG)


def test_duplicate_turn_id_is_rejected() -> None:
    invalid = _canonical_dialogue()
    duplicate = copy.deepcopy(invalid["turns"][0])
    invalid["turns"].append(duplicate)
    with pytest.raises(ValidationError, match="Duplicate turn_id"):
        validate_and_normalize(invalid, CONFIG)


def test_non_increasing_turn_id_is_rejected() -> None:
    invalid = _canonical_dialogue()
    earlier = copy.deepcopy(invalid["turns"][0])
    invalid["turns"][0]["turn_id"] = 2
    invalid["turns"].append(earlier)
    with pytest.raises(ValidationError, match="not in increasing order"):
        validate_and_normalize(invalid, CONFIG)


def test_unknown_speaker_is_rejected() -> None:
    invalid = _canonical_dialogue()
    invalid["turns"][0]["speaker"] = "doctor"
    with pytest.raises(ValidationError, match="has no configured voice"):
        validate_and_normalize(invalid, CONFIG)


def test_v0_2_fields_are_normalized_without_backend_mapping() -> None:
    dialogue = validate_and_normalize(
        _canonical_dialogue(
            {
                "rate": "slow",
                "pause_before_ms": 150,
                "pause_after_ms": 850,
                "arousal": "high",
                "coarse_affect": "distressed",
                "emotion": "fearful",
                "paralinguistic_events": ["sigh"],
            }
        ),
        CONFIG,
    )
    turn = dialogue.turns[0]
    assert turn.turn_id == 1
    assert turn.speaker == "caller"
    assert turn.text == "I need to talk."
    assert turn.label == "alert"
    assert turn.rate == "slow"
    assert turn.pause_before_ms == 150
    assert turn.pause_after_ms == 850
    assert turn.arousal == "high"
    assert turn.coarse_affect == "distressed"
    assert turn.emotion == "fearful"
    assert turn.paralinguistic_events == ["sigh"]


def test_v0_2_defaults_are_applied() -> None:
    turn = validate_and_normalize(_canonical_dialogue(), CONFIG).turns[0]
    assert turn.rate == "normal"
    assert turn.pause_before_ms == 0
    assert turn.pause_after_ms == CONFIG["pause"]["default_ms"]
    assert turn.arousal is None
    assert turn.coarse_affect is None
    assert turn.emotion is None
    assert turn.paralinguistic_events == []


def test_legacy_flat_fields_are_normalized() -> None:
    legacy = {
        "dialogue_id": "legacy001",
        "turns": [
            {
                "turn_id": 1,
                "speaker": "caller",
                "text": "Hello.",
                "label": "normal",
                "rate": "-10%",
                "pause_after_ms": 650,
                "arousal": "legacy-value",
                "emotion": None,
                "paralinguistic_events": [],
            }
        ],
    }
    turn = validate_and_normalize(legacy, CONFIG).turns[0]
    assert turn.rate == "-10%"
    assert turn.pause_before_ms == 0
    assert turn.pause_after_ms == 650
    assert turn.arousal == "legacy-value"
    assert turn.coarse_affect is None
    assert turn.emotion is None
    assert turn.paralinguistic_events == []


@pytest.mark.parametrize(
    "raw", [LEGACY_MINIMAL, _canonical_dialogue({"paralinguistic_events": ["sigh"]})]
)
def test_raw_input_is_not_mutated(raw: dict) -> None:
    before = copy.deepcopy(raw)
    validate_and_normalize(raw, CONFIG)
    assert raw == before


def test_canonical_and_legacy_fields_cannot_be_mixed() -> None:
    invalid = _canonical_dialogue()
    invalid["turns"][0]["rate"] = "+10%"
    with pytest.raises(ValidationError):
        validate_and_normalize(invalid, CONFIG)


def test_unsupported_schema_version_is_rejected() -> None:
    invalid = _canonical_dialogue()
    invalid["schema_version"] = "0.3"
    with pytest.raises(ValidationError):
        validate_and_normalize(invalid, CONFIG)


def test_kokoro_uses_its_own_voice_map() -> None:
    config = copy.deepcopy(CONFIG)
    config["tts"]["engine"] = "kokoro"
    dialogue = validate_and_normalize(LEGACY_MINIMAL, config)
    assert dialogue.turns[0].speaker == "counsellor"


def test_cosyvoice_uses_its_own_voice_map() -> None:
    config = copy.deepcopy(CONFIG)
    config["tts"]["engine"] = "cosyvoice"
    dialogue = validate_and_normalize(LEGACY_MINIMAL, config)
    assert dialogue.turns[1].speaker == "caller"


def test_cosyvoice_requires_complete_voice_entries() -> None:
    config = copy.deepcopy(CONFIG)
    del config["tts"]["cosyvoice"]["voice_map"]["caller"]["prompt_text"]
    with pytest.raises(ConfigError, match="caller.prompt_text"):
        validate_config(config)


def test_chatterbox_turbo_uses_project_speaker_map() -> None:
    config = copy.deepcopy(CONFIG)
    config["tts"]["engine"] = "chatterbox_turbo"
    dialogue = validate_and_normalize(LEGACY_MINIMAL, config)
    assert dialogue.turns[1].speaker == "caller"
