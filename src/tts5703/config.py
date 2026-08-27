"""Configuration loading for the rendering pipeline."""

from pathlib import Path
from typing import Any

import yaml

VALID_ENGINES = {"edge_tts", "kokoro", "chatterbox_turbo", "cosyvoice"}


class ConfigError(Exception):
    """Raised when config.yaml is structurally present but semantically invalid."""


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate a YAML pipeline configuration file."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ConfigError(
            f"Configuration file must contain a YAML mapping: {config_path}"
        )
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
        raise ConfigError(
            f"Missing required configuration key: pause.{error}"
        ) from error
    if not isinstance(default_pause, (int, float)) or default_pause < 0:
        raise ConfigError(
            f"pause.default_ms must be non-negative; got: {default_pause!r}"
        )

    try:
        tel = config["telephone"]
        sample_rate = tel["sample_rate"]
        high_pass = tel["high_pass_hz"]
        low_pass = tel["low_pass_hz"]
        channels = tel["channels"]
    except KeyError as error:
        raise ConfigError(
            f"Missing required configuration key: telephone.{error}"
        ) from error

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
        raise ConfigError(
            "tts.kokoro configuration is required when tts.engine is kokoro"
        )
    if engine == "chatterbox_turbo" and "chatterbox_turbo" not in config.get("tts", {}):
        raise ConfigError(
            "tts.chatterbox_turbo configuration is required when tts.engine is chatterbox_turbo"
        )
    if engine == "cosyvoice" and "cosyvoice" not in config.get("tts", {}):
        raise ConfigError(
            "tts.cosyvoice configuration is required when tts.engine is cosyvoice"
        )
    if engine == "cosyvoice":
        cosyvoice = config["tts"]["cosyvoice"]
        if not isinstance(cosyvoice, dict):
            raise ConfigError("tts.cosyvoice must be a mapping")
        voice_map = cosyvoice.get("voice_map")
        if not isinstance(voice_map, dict) or not voice_map:
            raise ConfigError("tts.cosyvoice.voice_map must be a non-empty mapping")
        for speaker, voice in voice_map.items():
            if not isinstance(voice, dict):
                raise ConfigError(
                    f"tts.cosyvoice.voice_map.{speaker} must be a mapping"
                )
            for field in ("prompt_wav", "prompt_text"):
                value = voice.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ConfigError(
                        f"tts.cosyvoice.voice_map.{speaker}.{field} "
                        "must be a non-empty string"
                    )
        for field in ("python_bin", "repo_dir", "model_dir"):
            value = cosyvoice.get(field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"tts.cosyvoice.{field} must be a non-empty string")
        for field in ("load_trt", "load_vllm", "fp16"):
            value = cosyvoice.get(field)
            if value is not None and not isinstance(value, bool):
                raise ConfigError(f"tts.cosyvoice.{field} must be true or false")
