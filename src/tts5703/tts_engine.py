"""Public TTS orchestration boundary and explicit backend dispatch."""

from pathlib import Path
from typing import Any

from .backends import cosyvoice_final, higgs
from .config import get_engine as _get_engine
from .render_plan import PreparedDialogue, TurnRenderResult


def get_engine(config: dict[str, Any]) -> str:
    """Return the explicitly selected supported engine."""
    return _get_engine(config)


def preflight_prepared_dialogue(
    dialogue: PreparedDialogue, config: dict[str, Any]
) -> None:
    """Validate every final prepared turn before starting any worker request."""
    engine = get_engine(config)
    if engine not in {"higgs", "cosyvoice"}:
        raise RuntimeError(
            f"Final prepared execution is not implemented for {engine!r}"
        )
    if dialogue.engine != engine:
        raise RuntimeError(
            f"Prepared dialogue engine {dialogue.engine!r} does not match "
            f"tts.engine {engine!r}"
        )
    if not dialogue.turns:
        raise RuntimeError("Prepared dialogue has no turns")
    ordinals = [turn.ordinal for turn in dialogue.turns]
    if ordinals != list(range(1, len(dialogue.turns) + 1)):
        raise RuntimeError(
            "Prepared dialogue ordinals must be unique, ordered, and 1-based"
        )
    for turn in dialogue.turns:
        if turn.backend != engine:
            raise RuntimeError(
                f"Prepared turn {turn.ordinal} backend does not match dialogue engine"
            )
        if engine == "higgs":
            higgs.preflight_prepared_turn(turn, config)
        else:
            cosyvoice_final.preflight_prepared_turn(turn, config)


async def synthesize_prepared_turns(
    dialogue: PreparedDialogue, out_dir: Path, config: dict[str, Any]
) -> tuple[TurnRenderResult, ...]:
    """Preflight the whole final dialogue, then render sequentially."""
    preflight_prepared_dialogue(dialogue, config)
    engine = get_engine(config)
    results: list[TurnRenderResult] = []
    for turn in dialogue.turns:
        if engine == "higgs":
            output_path = higgs.synthesize_prepared_turn(turn, out_dir, config)
            rate_status = "executed" if turn.rate_plan["enabled"] else "not_required"
        else:
            output_path = cosyvoice_final.synthesize_prepared_turn(
                turn, out_dir, config
            )
            rate_status = "executed"
        results.append(
            TurnRenderResult(
                ordinal=turn.ordinal,
                source_turn_id=turn.source_turn_id,
                output_path=output_path,
                synthesis_status="synthesized",
                rate_status=rate_status,
            )
        )
    return tuple(results)
