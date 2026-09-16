"""Public TTS orchestration boundary and explicit backend dispatch."""

from pathlib import Path
from typing import Any

from . import backend_info
from .backends import cosyvoice, higgs, kokoro
from .config import get_engine as _get_engine
from .cosyvoice_controls import BackendControlError
from .higgs_controls import HiggsControlError, resolve_higgs_turn_controls
from .validate import NormalizedTurn

# Stable project-level helper imports retained for scripts and tests.
build_cosyvoice_request = cosyvoice.build_request
rate_to_kokoro_speed = kokoro.rate_to_kokoro_speed


def get_engine(config: dict[str, Any]) -> str:
    """Return the explicitly selected supported engine."""
    return _get_engine(config)


def backend_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper for the backend identity public helper."""
    return backend_info.backend_identity(config)


def describe_engine(config: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper for the engine description public helper."""
    return backend_info.describe_engine(config)


def preflight_cosyvoice_controls(turn: NormalizedTurn) -> None:
    cosyvoice.preflight_controls(turn)


def preflight_higgs_controls(turn: NormalizedTurn) -> None:
    """Resolve one Higgs control plan without starting a model or worker."""
    try:
        resolve_higgs_turn_controls(turn)
    except HiggsControlError as error:
        raise BackendControlError(str(error)) from error


def preflight_backend_controls(turn: NormalizedTurn, config: dict[str, Any]) -> None:
    """Validate one turn against the selected backend's control mappings."""
    engine = get_engine(config)
    if engine == "cosyvoice":
        preflight_cosyvoice_controls(turn)
    elif engine == "higgs":
        preflight_higgs_controls(turn)


def preflight_dialogue_controls(
    turns: list[NormalizedTurn], config: dict[str, Any]
) -> None:
    """Fail a whole dialogue before spending any synthesis time on it."""
    for turn in turns:
        try:
            preflight_backend_controls(turn, config)
        except BackendControlError as error:
            raise BackendControlError(f"turn {turn.turn_id}: {error}") from error


async def synthesize_turn(
    turn: NormalizedTurn, out_dir: Path, config: dict[str, Any]
) -> Path:
    """Synthesize one turn with the engine selected in ``tts.engine``."""
    engine = get_engine(config)
    if engine == "higgs":
        return higgs.synthesize_turn(turn, out_dir, config)
    if engine == "cosyvoice":
        return cosyvoice.synthesize_turn(turn, out_dir, config)
    if engine == "kokoro":
        return kokoro.synthesize_turn(turn, out_dir, config)
    raise RuntimeError(f"Synthesis is not implemented for tts.engine {engine!r}")


async def synthesize_all_turns(
    turns: list[NormalizedTurn], out_dir: Path, config: dict[str, Any]
) -> dict[int, Path]:
    """Render one file per turn, sequentially in validated turn order."""
    return {
        turn.turn_id: await synthesize_turn(turn, out_dir, config) for turn in turns
    }
