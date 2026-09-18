"""Backend-specific control preflight, separate from canonical schema validation."""

import asyncio
import copy
import sys
from pathlib import Path

import pytest

import tts5703
from tts5703 import higgs_controls, tts_engine
from tts5703.backend_errors import BackendControlError as CanonicalBackendControlError
from tts5703.backends import cosyvoice as cosyvoice_backend
from tts5703.backends import higgs as higgs_backend
from tts5703.cosyvoice_controls import BackendControlError as CosyBackendControlError
from tts5703.tts_engine import (
    BackendControlError,
    preflight_backend_controls,
    preflight_cosyvoice_controls,
    preflight_dialogue_controls,
    preflight_higgs_controls,
)
from tts5703.validate import NormalizedTurn, validate_and_normalize

CONFIG_PATH = Path("config/config.yaml")
KOKORO_CONFIG_PATH = Path("config/config.kokoro.yaml")


def test_backend_modules_keep_package_worker_and_project_root_paths() -> None:
    package_dir = Path(tts5703.__file__).resolve().parent
    project_root = package_dir.parents[1]

    assert cosyvoice_backend._WORKER_SCRIPT == package_dir / "cosyvoice_worker.py"
    assert higgs_backend._WORKER_SCRIPT == package_dir / "higgs_worker.py"
    assert cosyvoice_backend._project_root() == project_root
    assert higgs_backend._project_root() == project_root


@pytest.fixture(scope="module")
def cosyvoice_config() -> dict:
    from tts5703.config import load_config

    config = copy.deepcopy(load_config(CONFIG_PATH))
    config["tts"]["engine"] = "cosyvoice"
    return config


@pytest.fixture(scope="module")
def kokoro_config() -> dict:
    from tts5703.config import load_config

    return load_config(KOKORO_CONFIG_PATH)


def _turn(
    *,
    turn_id: int = 1,
    rate: str = "normal",
    arousal: str | None = None,
    coarse_affect: str | None = None,
) -> NormalizedTurn:
    return NormalizedTurn(
        turn_id=turn_id,
        speaker="caller",
        text="I do not know what to do next.",
        label="alert",
        rate=rate,
        pause_before_ms=0,
        pause_after_ms=500,
        arousal=arousal,
        coarse_affect=coarse_affect,
    )


def _dialogue(coarse_affect: str | None) -> dict:
    return {
        "schema_version": "0.2",
        "dialogue_id": "preflight001",
        "turns": [
            {
                "turn_id": 1,
                "speaker": "caller",
                "text": "I do not know what to do next.",
                "label": "alert",
                "acoustic_spec": {"coarse_affect": coarse_affect},
            }
        ],
    }


def test_canonical_schema_still_accepts_open_coarse_affect_values(
    cosyvoice_config: dict,
) -> None:
    dialogue = validate_and_normalize(_dialogue("anxious"), cosyvoice_config)

    assert dialogue.turns[0].coarse_affect == "anxious"


def test_cosyvoice_preflight_rejects_unmapped_coarse_affect() -> None:
    with pytest.raises(
        BackendControlError,
        match=(
            r"Unsupported CosyVoice coarse_affect mapping: 'cheerful'\. "
            r"Currently supported mappings: neutral, sad, anxious, angry, warm, "
            r"distressed\."
        ),
    ):
        preflight_cosyvoice_controls(_turn(coarse_affect="cheerful"))


def test_cosyvoice_preflight_rejects_unmapped_arousal() -> None:
    with pytest.raises(
        BackendControlError,
        match=(
            r"Unsupported CosyVoice arousal mapping: 'frantic'\. "
            r"Currently supported mappings: low, medium, high\."
        ),
    ):
        preflight_cosyvoice_controls(_turn(arousal="frantic"))


@pytest.mark.parametrize(
    "coarse_affect",
    ["neutral", "sad", "anxious", "angry", "warm", "distressed", None],
)
def test_cosyvoice_preflight_accepts_mapped_coarse_affect(
    coarse_affect: str | None,
) -> None:
    assert preflight_cosyvoice_controls(_turn(coarse_affect=coarse_affect)) is None


@pytest.mark.parametrize("arousal", ["low", "medium", "high", None])
def test_cosyvoice_preflight_accepts_mapped_arousal(arousal: str | None) -> None:
    assert preflight_cosyvoice_controls(_turn(arousal=arousal)) is None


def test_kokoro_does_not_reject_schema_valid_ignored_controls(
    kokoro_config: dict,
) -> None:
    dialogue = validate_and_normalize(_dialogue("anxious"), kokoro_config)

    assert dialogue.turns[0].coarse_affect == "anxious"
    assert preflight_backend_controls(dialogue.turns[0], kokoro_config) is None
    assert preflight_backend_controls(_turn(arousal="frantic"), kokoro_config) is None


def test_higgs_preflight_delegates_exact_turn_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def resolve(**kwargs: object) -> dict:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(higgs_controls, "resolve_higgs_controls", resolve)
    turn = _turn(rate="slow", arousal="low", coarse_affect="warm")

    assert preflight_higgs_controls(turn) is None
    assert captured == {
        "text": turn.text,
        "rate": "slow",
        "arousal": "low",
        "coarse_affect": "warm",
        "pause_before_ms": 0,
        "pause_after_ms": 500,
        "speaker_id": "caller",
    }


@pytest.mark.parametrize(
    ("arousal", "coarse_affect"),
    [(None, None), ("low", "warm")],
)
def test_higgs_preflight_accepts_supported_controls(
    arousal: str | None, coarse_affect: str | None
) -> None:
    config = {"tts": {"engine": "higgs"}}

    assert (
        preflight_backend_controls(
            _turn(arousal=arousal, coarse_affect=coarse_affect), config
        )
        is None
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("arousal", "frantic"), ("coarse_affect", "cheerful")],
)
def test_higgs_preflight_rejects_invalid_non_null_controls(
    field: str, value: str
) -> None:
    config = {"tts": {"engine": "higgs"}}

    with pytest.raises(BackendControlError, match=rf"Unsupported Higgs {field}"):
        preflight_backend_controls(_turn(**{field: value}), config)


@pytest.mark.parametrize("rate", ["+10%", "-20%"])
def test_higgs_preflight_rejects_legacy_percentage_rates(rate: str) -> None:
    config = {"tts": {"engine": "higgs"}}

    with pytest.raises(BackendControlError, match="Unsupported Higgs rate mapping"):
        preflight_backend_controls(_turn(rate=rate), config)


def test_higgs_preflight_has_no_worker_or_sglang_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {"tts": {"engine": "higgs"}}

    def unreachable_worker(*args: object) -> object:
        raise AssertionError("preflight must remain CPU-only")

    monkeypatch.setattr(higgs_backend, "_get_worker", unreachable_worker)

    assert preflight_backend_controls(_turn(), config) is None
    assert "sglang" not in sys.modules


def test_higgs_missing_reference_fails_before_worker_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {
        "tts": {
            "engine": "higgs",
            "higgs": {
                "server_executable": "missing-sgl-omni",
                "model_dir": "missing-model",
                "voice_map": {
                    "caller": {"reference_wav": str(tmp_path / "missing.wav")}
                },
            },
        }
    }

    def unreachable_worker(*args: object) -> object:
        raise AssertionError("missing reference must fail before worker startup")

    monkeypatch.setattr(higgs_backend, "_get_worker", unreachable_worker)

    with pytest.raises(RuntimeError, match="Higgs reference audio is missing"):
        asyncio.run(tts_engine.synthesize_turn(_turn(), tmp_path, config))

    assert list(tmp_path.iterdir()) == []


def test_dialogue_preflight_reports_the_offending_turn(
    cosyvoice_config: dict,
) -> None:
    turns = [_turn(turn_id=1), _turn(turn_id=2, coarse_affect="cheerful")]

    with pytest.raises(BackendControlError, match=r"turn 2: Unsupported CosyVoice"):
        preflight_dialogue_controls(turns, cosyvoice_config)


def test_dialogue_preflight_accepts_mapped_cosyvoice_controls(
    cosyvoice_config: dict,
) -> None:
    turns = [
        _turn(turn_id=1, arousal="low", coarse_affect="neutral"),
        _turn(turn_id=2, arousal="high", coarse_affect="distressed"),
    ]

    assert preflight_dialogue_controls(turns, cosyvoice_config) is None


def test_cosyvoice_synthesis_fails_preflight_before_reaching_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cosyvoice_config: dict
) -> None:
    config = copy.deepcopy(cosyvoice_config)
    worker_calls: list[dict] = []

    def unreachable_worker(*args: object) -> object:
        worker_calls.append({"args": args})
        raise AssertionError("preflight must run before the worker starts")

    monkeypatch.setattr(cosyvoice_backend, "_get_worker", unreachable_worker)

    with pytest.raises(BackendControlError, match="coarse_affect mapping: 'cheerful'"):
        asyncio.run(
            tts_engine.synthesize_turn(
                _turn(coarse_affect="cheerful"), tmp_path, config
            )
        )

    assert worker_calls == []
    assert list(tmp_path.iterdir()) == []


def test_backend_control_error_stays_catchable_as_value_error() -> None:
    assert issubclass(BackendControlError, ValueError)


def test_backend_control_error_exports_share_one_class() -> None:
    assert CanonicalBackendControlError is CosyBackendControlError
    assert CanonicalBackendControlError is BackendControlError
