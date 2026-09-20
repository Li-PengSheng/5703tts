"""Load one explicit backend selection and shared audio configuration.

``tts.engine`` selects Higgs or CosyVoice for the complete run. Validation does
not implement automatic fallback or rewrite defaults: ``config/config.yaml``
selects production-primary Higgs, while ``config/config_cosyvoice.yaml`` is the
explicit backup/secondary configuration.
"""

from pathlib import Path
from typing import Any

import yaml

VALID_ENGINES = {"cosyvoice", "higgs"}


class ConfigError(Exception):
    """Raised when config.yaml is structurally present but semantically invalid."""


def get_engine(config: dict[str, Any]) -> str:
    """Return the one explicitly selected engine; never infer a fallback."""
    try:
        engine = config["tts"]["engine"]
    except (KeyError, TypeError) as error:
        raise ValueError("Missing required configuration key: tts.engine") from error
    if engine not in VALID_ENGINES:
        raise ValueError(
            f"Unsupported tts.engine: {engine} (available: {sorted(VALID_ENGINES)})"
        )
    return engine


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML and validate only the selected production runtime contract."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ConfigError(
            f"Configuration file must contain a YAML mapping: {config_path}"
        )
    _validate_config(config)
    return config


def _validate_higgs(higgs: Any) -> None:
    """Validate the selected Higgs runtime contract."""
    if not isinstance(higgs, dict):
        raise ConfigError("tts.higgs must be a mapping")

    for field in ("server_executable", "model_dir"):
        value = higgs.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"tts.higgs.{field} must be a non-empty string")

    host = higgs.get("host", "127.0.0.1")
    if host != "127.0.0.1":
        raise ConfigError("tts.higgs.host must be 127.0.0.1")

    port = higgs.get("port", 18080)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ConfigError("tts.higgs.port must be an integer from 1 to 65535")

    for field, default in (
        ("startup_timeout_seconds", 900),
        ("inference_timeout_seconds", 300),
    ):
        value = higgs.get(field, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"tts.higgs.{field} must be a positive number")

    ffmpeg_bin = higgs.get("ffmpeg_bin", "ffmpeg")
    if not isinstance(ffmpeg_bin, str) or not ffmpeg_bin.strip():
        raise ConfigError("tts.higgs.ffmpeg_bin must be a non-empty string")


def _validate_config(config: dict[str, Any]) -> None:
    """Catch obviously-broken config early instead of failing deep inside a batch run."""
    tts = config.get("tts")
    if not isinstance(tts, dict):
        raise ConfigError("tts must be a mapping")
    engine = tts.get("engine")
    if engine not in VALID_ENGINES:
        raise ConfigError(f"tts.engine must be one of {VALID_ENGINES}; got: {engine!r}")

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

    if engine == "cosyvoice" and "cosyvoice" not in tts:
        raise ConfigError(
            "tts.cosyvoice configuration is required when tts.engine is cosyvoice"
        )
    if engine == "cosyvoice":
        cosyvoice = tts["cosyvoice"]
        if not isinstance(cosyvoice, dict):
            raise ConfigError("tts.cosyvoice must be a mapping")
        for field in ("python_bin", "repo_dir", "model_dir"):
            value = cosyvoice.get(field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"tts.cosyvoice.{field} must be a non-empty string")
        for field in ("load_trt", "load_vllm", "fp16"):
            value = cosyvoice.get(field)
            if value is not None and not isinstance(value, bool):
                raise ConfigError(f"tts.cosyvoice.{field} must be true or false")
        # Optional: pins the sample rate recorded in metadata when the worker's
        # runtime value cannot be observed.
        sample_rate = cosyvoice.get("sample_rate")
        if sample_rate is not None and (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate <= 0
        ):
            raise ConfigError(
                "tts.cosyvoice.sample_rate must be a positive integer; "
                f"got: {sample_rate!r}"
            )
    if engine == "higgs" and "higgs" not in tts:
        raise ConfigError(
            "tts.higgs configuration is required when tts.engine is higgs"
        )
    if engine == "higgs":
        _validate_higgs(tts["higgs"])
