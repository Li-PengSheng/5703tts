"""Early Kokoro config validation and shared-configuration drift guard."""

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from tts5703.config import ConfigError, _validate_config, load_config
from tts5703.validate import ValidationError, validate_and_normalize

CONFIG_PATH = Path("config/config.yaml")
KOKORO_CONFIG_PATH = Path("config/config.kokoro.yaml")

# config.yaml selects CosyVoice and additionally configures EdgeTTS, Chatterbox
# Turbo, and CosyVoice; config.kokoro.yaml only carries what the Kokoro
# controlled-benchmark baseline needs. Everything below must stay identical so a
# Kokoro run means the same thing under either file.
SHARED_KOKORO_FIELDS = (
    ("tts", "kokoro"),
    ("tts", "default_rate"),
    ("pause",),
    ("fade_ms",),
    ("telephone",),
)
# Intentional differences, deliberately excluded from the guard above.
KNOWN_DIFFERENCES = (("tts", "engine"),)


def _kokoro_config() -> dict[str, Any]:
    config = copy.deepcopy(load_config(KOKORO_CONFIG_PATH))
    return config


def _at(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        value = value[key]
    return value


def test_reference_kokoro_config_is_valid() -> None:
    assert load_config(KOKORO_CONFIG_PATH)["tts"]["engine"] == "kokoro"


@pytest.mark.parametrize("lang_code", [None, "", "   ", 42, ["a"]])
def test_invalid_lang_code_fails_early(lang_code: Any) -> None:
    config = _kokoro_config()
    config["tts"]["kokoro"]["lang_code"] = lang_code

    with pytest.raises(ConfigError, match="tts.kokoro.lang_code must be a non-empty"):
        _validate_config(config)


def test_missing_lang_code_fails_early() -> None:
    config = _kokoro_config()
    del config["tts"]["kokoro"]["lang_code"]

    with pytest.raises(ConfigError, match="tts.kokoro.lang_code"):
        _validate_config(config)


@pytest.mark.parametrize("sample_rate", [None, 0, -24_000, 24_000.5, "24000", True])
def test_invalid_sample_rate_fails_early(sample_rate: Any) -> None:
    config = _kokoro_config()
    config["tts"]["kokoro"]["sample_rate"] = sample_rate

    with pytest.raises(
        ConfigError, match="tts.kokoro.sample_rate must be a positive integer"
    ):
        _validate_config(config)


@pytest.mark.parametrize("voice_map", [None, {}, [], "af_heart"])
def test_invalid_voice_map_fails_early(voice_map: Any) -> None:
    config = _kokoro_config()
    config["tts"]["kokoro"]["voice_map"] = voice_map

    with pytest.raises(
        ConfigError, match="tts.kokoro.voice_map must be a non-empty mapping"
    ):
        _validate_config(config)


@pytest.mark.parametrize("role", ["caller", "counsellor"])
def test_partial_voice_map_is_a_dialogue_concern_not_a_config_error(role: str) -> None:
    """Which speakers must exist depends on the dialogue, not on the backend."""
    config = _kokoro_config()
    del config["tts"]["kokoro"]["voice_map"][role]

    assert _validate_config(config) is None


def test_voice_map_for_unrelated_speaker_roles_is_accepted() -> None:
    config = _kokoro_config()
    config["tts"]["kokoro"]["voice_map"] = {"narrator": "af_heart"}

    assert _validate_config(config) is None


def test_unused_missing_speaker_mapping_still_fails_at_dialogue_validation() -> None:
    config = _kokoro_config()
    del config["tts"]["kokoro"]["voice_map"]["caller"]
    dialogue = {
        "schema_version": "0.2",
        "dialogue_id": "roles001",
        "turns": [
            {
                "turn_id": 1,
                "speaker": "caller",
                "text": "I need to talk.",
                "label": "alert",
                "acoustic_spec": {},
            }
        ],
    }

    with pytest.raises(ValidationError, match="has no configured voice"):
        validate_and_normalize(dialogue, config)


@pytest.mark.parametrize("speaker", ["", "   ", 7])
def test_invalid_voice_map_key_fails_early(speaker: Any) -> None:
    config = _kokoro_config()
    config["tts"]["kokoro"]["voice_map"] = {speaker: "af_heart"}

    with pytest.raises(ConfigError, match="voice_map keys must be non-empty strings"):
        _validate_config(config)


@pytest.mark.parametrize("voice", [None, "", "  ", 7])
def test_empty_speaker_voice_fails_early(voice: Any) -> None:
    config = _kokoro_config()
    config["tts"]["kokoro"]["voice_map"]["caller"] = voice

    with pytest.raises(ConfigError, match="tts.kokoro.voice_map.caller"):
        _validate_config(config)


def test_non_mapping_kokoro_section_fails_early() -> None:
    config = _kokoro_config()
    config["tts"]["kokoro"] = "kokoro"

    with pytest.raises(ConfigError, match="tts.kokoro must be a mapping"):
        _validate_config(config)


@pytest.mark.parametrize("device", ["", "   ", 0])
def test_invalid_device_fails_early(device: Any) -> None:
    config = _kokoro_config()
    config["tts"]["kokoro"]["device"] = device

    with pytest.raises(ConfigError, match="tts.kokoro.device must be null"):
        _validate_config(config)


@pytest.mark.parametrize("device", [None, "cpu"])
def test_supported_device_values_are_accepted(device: Any) -> None:
    config = _kokoro_config()
    config["tts"]["kokoro"]["device"] = device

    assert _validate_config(config) is None


def test_kokoro_validation_does_not_apply_to_other_engines() -> None:
    config = copy.deepcopy(load_config(CONFIG_PATH))
    config["tts"]["kokoro"]["lang_code"] = ""

    assert config["tts"]["engine"] != "kokoro"
    assert _validate_config(config) is None


@pytest.mark.parametrize("path", SHARED_KOKORO_FIELDS, ids=lambda path: ".".join(path))
def test_shared_kokoro_configuration_stays_equivalent(path: tuple[str, ...]) -> None:
    default = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    kokoro = yaml.safe_load(KOKORO_CONFIG_PATH.read_text(encoding="utf-8"))

    assert _at(default, path) == _at(kokoro, path), (
        f"{'.'.join(path)} drifted between {CONFIG_PATH} and {KOKORO_CONFIG_PATH}"
    )


def test_known_configuration_differences_are_still_the_only_ones() -> None:
    default = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    kokoro = yaml.safe_load(KOKORO_CONFIG_PATH.read_text(encoding="utf-8"))

    assert KNOWN_DIFFERENCES == (("tts", "engine"),)
    assert _at(default, ("tts", "engine")) == "cosyvoice"
    assert _at(kokoro, ("tts", "engine")) == "kokoro"
    # config.kokoro.yaml intentionally omits the other backends' sections.
    assert set(kokoro["tts"]) - {"engine", "default_rate", "kokoro"} == set()
