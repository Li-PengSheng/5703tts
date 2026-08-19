"""Fast validation boundary checks; no TTS/network required."""

import importlib
import copy
from pathlib import Path

config_module = importlib.import_module("tts5703.config")
validate_module = importlib.import_module("tts5703.validate")
load_config = config_module.load_config
ValidationError = validate_module.ValidationError
validate_and_normalize = validate_module.validate_and_normalize

CONFIG = load_config(Path("config/config.yaml"))
VALID = {
    "dialogue_id": "ok001",
    "turns": [
        {"turn_id": 1, "speaker": "counsellor", "text": "Hi.", "label": "normal"},
        {"turn_id": 2, "speaker": "caller", "text": "Hello.", "label": "normal"},
    ],
}


def test_valid_dialogue_is_normalized() -> None:
    dialogue = validate_and_normalize(VALID, CONFIG)
    assert dialogue.dialogue_id == "ok001"
    assert dialogue.turns[0].rate == "+0%"
    assert dialogue.turns[0].pause_after_ms == 500


def test_invalid_label_is_rejected() -> None:
    invalid = {
        "dialogue_id": "bad",
        "turns": [
            {"turn_id": 1, "speaker": "caller", "text": "Hello.", "label": "urgent"}
        ],
    }
    try:
        validate_and_normalize(invalid, CONFIG)
    except ValidationError:
        return
    raise AssertionError("invalid label should be rejected")


def test_unknown_speaker_is_rejected() -> None:
    invalid = {
        "dialogue_id": "bad",
        "turns": [
            {"turn_id": 1, "speaker": "doctor", "text": "Hello.", "label": "normal"}
        ],
    }
    try:
        validate_and_normalize(invalid, CONFIG)
    except ValidationError:
        return
    raise AssertionError("unknown speaker should be rejected")


def test_kokoro_uses_its_own_voice_map() -> None:
    config = copy.deepcopy(CONFIG)
    config["tts"]["engine"] = "kokoro"
    dialogue = validate_and_normalize(VALID, config)
    assert dialogue.turns[0].speaker == "counsellor"


def test_chatterbox_turbo_uses_project_speaker_map() -> None:
    config = copy.deepcopy(CONFIG)
    config["tts"]["engine"] = "chatterbox_turbo"
    dialogue = validate_and_normalize(VALID, config)
    assert dialogue.turns[1].speaker == "caller"
