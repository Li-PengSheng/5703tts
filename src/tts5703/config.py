"""Configuration loading for the rendering pipeline."""

from pathlib import Path
from typing import Any

import yaml

VALID_ENGINES = {"edge_tts", "kokoro", "chatterbox_turbo"}


class ConfigError(Exception):
    """Raised when config.yaml is structurally present but semantically invalid."""


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate a YAML pipeline configuration file."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ConfigError(f"Configuration file must contain a YAML mapping: {config_path}")
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    """Catch obviously-broken config early instead of failing deep inside a batch run."""
    try:
        engine = config["tts"]["engine"]
    except KeyError as error:
        raise ConfigError(f"Missing required configuration key: tts.{error}") from error
    if engine not in VALID_ENGINES:
        raise ConfigError(f"tts.engine must be one of {VALID_ENGINES}; got: {engine!r}")

    try:
        default_pause = config["pause"]["default_ms"]
    except KeyError as error:
        raise ConfigError(f"Missing required configuration key: pause.{error}") from error
    if not isinstance(default_pause, (int, float)) or default_pause < 0:
        raise ConfigError(f"pause.default_ms must be non-negative; got: {default_pause!r}")

    try:
        tel = config["telephone"]
        sample_rate = tel["sample_rate"]
        high_pass = tel["high_pass_hz"]
        low_pass = tel["low_pass_hz"]
        channels = tel["channels"]
    except KeyError as error:
        raise ConfigError(f"Missing required configuration key: telephone.{error}") from error

    for name, value in [
        ("sample_rate", sample_rate),
        ("high_pass_hz", high_pass),
        ("low_pass_hz", low_pass),
        ("channels", channels),
    ]:
        if not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"telephone.{name} must be positive; got: {value!r}")

    if not (high_pass < low_pass):
        raise ConfigError(
            f"telephone.high_pass_hz ({high_pass}) must be less than low_pass_hz ({low_pass})"
        )
    nyquist = sample_rate / 2
    if not (low_pass < nyquist):
        raise ConfigError(
            f"telephone.low_pass_hz ({low_pass}) must be below the Nyquist frequency ({nyquist}), "
            f"half the sample rate; otherwise the filter configuration is invalid"
        )

    if engine == "kokoro" and "kokoro" not in config.get("tts", {}):
        raise ConfigError("tts.kokoro configuration is required when tts.engine is kokoro")
    if engine == "chatterbox_turbo" and "chatterbox_turbo" not in config.get("tts", {}):
        raise ConfigError(
            "tts.chatterbox_turbo configuration is required when tts.engine is chatterbox_turbo"
        )
