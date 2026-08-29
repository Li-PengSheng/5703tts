"""Backend-specific control preflight, separate from canonical schema validation."""

import asyncio
import copy
from pathlib import Path

import pytest

from tts5703 import tts_engine
from tts5703.tts_engine import (
    BackendControlError,
    preflight_backend_controls,
    preflight_cosyvoice_controls,
    preflight_dialogue_controls,
)
from tts5703.validate import NormalizedTurn, validate_and_normalize

CONFIG_PATH = Path("config/config.yaml")
KOKORO_CONFIG_PATH = Path("config/config.kokoro.yaml")


@pytest.fixture(scope="module")
def cosyvoice_config() -> dict:
    from tts5703.config import load_config

    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def kokoro_config() -> dict:
    from tts5703.config import load_config

    return load_config(KOKORO_CONFIG_PATH)


def _turn(
    *,
    turn_id: int = 1,
    arousal: str | None = None,
    coarse_affect: str | None = None,
) -> NormalizedTurn:
    return NormalizedTurn(
        turn_id=turn_id,
        speaker="caller",
        text="I do not know what to do next.",
        label="alert",
        rate="normal",
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
            r"Unsupported CosyVoice coarse_affect mapping: 'anxious'\. "
            r"Currently supported mappings: neutral, distressed\."
        ),
    ):
        preflight_cosyvoice_controls(_turn(coarse_affect="anxious"))


def test_cosyvoice_preflight_rejects_unmapped_arousal() -> None:
    with pytest.raises(
        BackendControlError,
        match=(
            r"Unsupported CosyVoice arousal mapping: 'frantic'\. "
            r"Currently supported mappings: low, medium, high\."
        ),
    ):
        preflight_cosyvoice_controls(_turn(arousal="frantic"))


@pytest.mark.parametrize("coarse_affect", ["neutral", "distressed", None])
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


def test_dialogue_preflight_reports_the_offending_turn(
    cosyvoice_config: dict,
) -> None:
    turns = [_turn(turn_id=1), _turn(turn_id=2, coarse_affect="anxious")]

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

    monkeypatch.setattr(tts_engine, "_get_cosyvoice_worker", unreachable_worker)

    with pytest.raises(BackendControlError, match="coarse_affect mapping: 'anxious'"):
        asyncio.run(
            tts_engine.synthesize_turn(_turn(coarse_affect="anxious"), tmp_path, config)
        )

    assert worker_calls == []
    assert list(tmp_path.iterdir()) == []


def test_backend_control_error_stays_catchable_as_value_error() -> None:
    assert issubclass(BackendControlError, ValueError)
