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
        raise ConfigError(f"配置文件必须是 YAML 对象: {config_path}")
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    """Catch obviously-broken config early instead of failing deep inside a batch run."""
    try:
        engine = config["tts"]["engine"]
    except KeyError as error:
        raise ConfigError(f"缺少必要配置项: tts.{error}") from error
    if engine not in VALID_ENGINES:
        raise ConfigError(f"tts.engine 必须是 {VALID_ENGINES} 之一，当前是: {engine!r}")

    try:
        default_pause = config["pause"]["default_ms"]
    except KeyError as error:
        raise ConfigError(f"缺少必要配置项: pause.{error}") from error
    if not isinstance(default_pause, (int, float)) or default_pause < 0:
        raise ConfigError(f"pause.default_ms 必须是非负数，当前是: {default_pause!r}")

    try:
        tel = config["telephone"]
        sample_rate = tel["sample_rate"]
        high_pass = tel["high_pass_hz"]
        low_pass = tel["low_pass_hz"]
        channels = tel["channels"]
    except KeyError as error:
        raise ConfigError(f"缺少必要配置项: telephone.{error}") from error

    for name, value in [
        ("sample_rate", sample_rate),
        ("high_pass_hz", high_pass),
        ("low_pass_hz", low_pass),
        ("channels", channels),
    ]:
        if not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"telephone.{name} 必须是正数，当前是: {value!r}")

    if not (high_pass < low_pass):
        raise ConfigError(
            f"telephone.high_pass_hz ({high_pass}) 必须小于 low_pass_hz ({low_pass})"
        )
    nyquist = sample_rate / 2
    if not (low_pass < nyquist):
        raise ConfigError(
            f"telephone.low_pass_hz ({low_pass}) 必须小于采样率一半的奈奎斯特频率 ({nyquist})，"
            f"否则滤波器设置无意义"
        )

    if engine == "kokoro" and "kokoro" not in config.get("tts", {}):
        raise ConfigError("tts.engine 设为 kokoro 时，必须提供 tts.kokoro 配置块")
    if engine == "chatterbox_turbo" and "chatterbox_turbo" not in config.get("tts", {}):
        raise ConfigError(
            "tts.engine 设为 chatterbox_turbo 时，必须提供 tts.chatterbox_turbo 配置块"
        )
