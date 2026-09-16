"""Kokoro model loading and turn synthesis."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..validate import NormalizedTurn

logger = logging.getLogger("tts5703.tts_engine")

_SEMANTIC_RATES = {
    "slow": 0.8,
    "normal": 1.0,
    "fast": 1.2,
}


def rate_to_kokoro_speed(rate: str) -> float:
    """Map semantic or legacy percentage rates to a Kokoro speed multiplier."""
    if rate in _SEMANTIC_RATES:
        return _SEMANTIC_RATES[rate]
    return max(0.1, 1 + int(rate[:-1]) / 100)


@lru_cache(maxsize=4)
def _get_kokoro_pipeline(lang_code: str, device: str | None):
    # Load Kokoro only when its backend is selected.
    from kokoro import KPipeline

    return KPipeline(lang_code=lang_code, device=device)


def synthesize_turn(
    turn: NormalizedTurn, out_dir: Path, config: dict[str, Any]
) -> Path:
    kokoro = config["tts"]["kokoro"]
    voice = kokoro["voice_map"][turn.speaker]
    output_path = out_dir / f"turn_{turn.turn_id:03d}.wav"
    logger.debug(
        "event=turn_tts_start engine=kokoro turn=%d speaker=%s voice=%s rate=%s",
        turn.turn_id,
        turn.speaker,
        voice,
        turn.rate,
    )
    pipeline = _get_kokoro_pipeline(kokoro["lang_code"], kokoro.get("device"))
    chunks = [
        np.asarray(result.audio)
        for result in pipeline(
            turn.text,
            voice=voice,
            speed=rate_to_kokoro_speed(turn.rate),
        )
    ]
    if not chunks:
        raise RuntimeError(f"Kokoro did not generate audio for turn {turn.turn_id}")
    sf.write(output_path, np.concatenate(chunks), kokoro["sample_rate"])
    logger.debug(
        "event=turn_tts_complete engine=kokoro turn=%d output=%s chunks=%d bytes=%d",
        turn.turn_id,
        output_path.name,
        len(chunks),
        output_path.stat().st_size,
    )
    return output_path
