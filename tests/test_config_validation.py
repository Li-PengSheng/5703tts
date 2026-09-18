from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tts5703.config import VALID_ENGINES, ConfigError, _validate_config, load_config

HIGGS_CONFIG = Path("config/config.yaml")
COSYVOICE_CONFIG = Path("config/config_cosyvoice.yaml")
HIGGS_EXAMPLE = Path("config/config.higgs.example.yaml")


def _higgs() -> dict[str, Any]:
    return deepcopy(load_config(HIGGS_CONFIG))


def _cosyvoice() -> dict[str, Any]:
    return deepcopy(load_config(COSYVOICE_CONFIG))


def test_production_engine_universe_is_exact() -> None:
    assert VALID_ENGINES == {"higgs", "cosyvoice"}


def test_checked_in_production_configs_are_valid() -> None:
    assert load_config(HIGGS_CONFIG)["tts"]["engine"] == "higgs"
    assert load_config(COSYVOICE_CONFIG)["tts"]["engine"] == "cosyvoice"
    assert load_config(HIGGS_EXAMPLE)["tts"]["engine"] == "higgs"


@pytest.mark.parametrize("config_factory", [_higgs, _cosyvoice])
def test_selected_backend_needs_no_voice_map(config_factory: Any) -> None:
    config = config_factory()
    selected = config["tts"]["engine"]

    assert "voice_map" not in config["tts"][selected]
    assert _validate_config(config) is None


def test_higgs_only_config_is_valid() -> None:
    config = _higgs()
    config["tts"].pop("cosyvoice")
    assert _validate_config(config) is None


def test_cosyvoice_only_config_is_valid() -> None:
    config = _cosyvoice()
    assert set(config["tts"]) == {"engine", "cosyvoice"}
    assert _validate_config(config) is None


@pytest.mark.parametrize("engine", ["higgs", "cosyvoice"])
def test_malformed_unselected_backend_is_ignored(engine: str) -> None:
    config = _higgs() if engine == "higgs" else _cosyvoice()
    config["tts"]["engine"] = engine
    config["tts"]["cosyvoice" if engine == "higgs" else "higgs"] = "invalid"
    assert _validate_config(config) is None


@pytest.mark.parametrize("engine", ["kokoro", "unknown", None])
def test_unsupported_or_missing_engine_is_rejected(engine: Any) -> None:
    config = _higgs()
    config["tts"]["engine"] = engine
    with pytest.raises(ConfigError, match="tts.engine must be one of"):
        _validate_config(config)


@pytest.mark.parametrize("field", ["server_executable", "model_dir"])
@pytest.mark.parametrize("value", [None, "", "  ", 7])
def test_selected_higgs_requires_runtime_paths(field: str, value: Any) -> None:
    config = _higgs()
    config["tts"]["higgs"][field] = value
    with pytest.raises(ConfigError, match=rf"tts.higgs.{field}"):
        _validate_config(config)


@pytest.mark.parametrize("host", ["localhost", "::1", "0.0.0.0", None])
def test_higgs_host_is_frozen_to_loopback(host: Any) -> None:
    config = _higgs()
    config["tts"]["higgs"]["host"] = host
    with pytest.raises(ConfigError, match="tts.higgs.host"):
        _validate_config(config)


@pytest.mark.parametrize("field", ["python_bin", "repo_dir", "model_dir"])
@pytest.mark.parametrize("value", ["", "  ", 7])
def test_selected_cosyvoice_runtime_paths_are_nonblank(field: str, value: Any) -> None:
    config = _cosyvoice()
    config["tts"]["cosyvoice"][field] = value
    with pytest.raises(ConfigError, match=rf"tts.cosyvoice.{field}"):
        _validate_config(config)


@pytest.mark.parametrize("field", ["load_trt", "load_vllm", "fp16"])
def test_selected_cosyvoice_flags_are_boolean(field: str) -> None:
    config = _cosyvoice()
    config["tts"]["cosyvoice"][field] = "yes"
    with pytest.raises(ConfigError, match=rf"tts.cosyvoice.{field}"):
        _validate_config(config)


def test_selected_backend_block_is_required() -> None:
    config = _higgs()
    del config["tts"]["higgs"]
    with pytest.raises(ConfigError, match="tts.higgs configuration is required"):
        _validate_config(config)


@pytest.mark.parametrize(
    "field", ["sample_rate", "channels", "high_pass_hz", "low_pass_hz"]
)
def test_shared_telephone_fields_must_be_positive(field: str) -> None:
    config = _higgs()
    config["telephone"][field] = 0
    with pytest.raises(ConfigError, match=rf"telephone.{field}"):
        _validate_config(config)
