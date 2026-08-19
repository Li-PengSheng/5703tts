"""Stage 2 TTS adapter, selectable through ``tts.engine``."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .validate import NormalizedTurn

logger = logging.getLogger(__name__)


def get_engine(config: dict[str, Any]) -> str:
    engine = config["tts"].get("engine", "edge_tts")
    if engine not in {"edge_tts", "kokoro", "chatterbox_turbo"}:
        raise ValueError(
            f"Unsupported tts.engine: {engine} (available: edge_tts, kokoro, chatterbox_turbo)"
        )
    return engine


def turn_audio_extension(config: dict[str, Any]) -> str:
    return ".mp3" if get_engine(config) == "edge_tts" else ".wav"


def _rate_to_kokoro_speed(rate: str) -> float:
    """Convert the existing EdgeTTS-style ``+/-N%`` setting to Kokoro speed."""
    return max(0.1, 1 + int(rate[:-1]) / 100)


@lru_cache(maxsize=4)
def _get_kokoro_pipeline(lang_code: str, device: str | None):
    # Lazy import keeps EdgeTTS runs independent of Kokoro model loading.
    from kokoro import KPipeline

    return KPipeline(lang_code=lang_code, device=device)


@lru_cache(maxsize=2)
def _get_chatterbox_turbo(device: str, model_dir: str | None, nano: bool):
    """Load Turbo/Nano lazily so EdgeTTS/Kokoro runs do not import its model stack."""
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    if model_dir:
        return ChatterboxTurboTTS.from_local(model_dir, device=device)
    return ChatterboxTurboTTS.from_pretrained(device=device, nano=nano)


async def synthesize_turn(
    turn: NormalizedTurn, out_dir: Path, config: dict[str, Any]
) -> Path:
    """Synthesize one turn with the engine selected in ``tts.engine``."""
    engine = get_engine(config)
    if engine == "edge_tts":
        import edge_tts  # Lazy import: only needed when Edge TTS is selected.

        voice = config["speaker_voice_map"][turn.speaker]
        output_path = out_dir / f"turn_{turn.turn_id:03d}.mp3"
        logger.debug(
            "event=turn_tts_start engine=edge_tts turn=%d speaker=%s voice=%s rate=%s",
            turn.turn_id,
            turn.speaker,
            voice,
            turn.rate,
        )
        await edge_tts.Communicate(text=turn.text, voice=voice, rate=turn.rate).save(
            str(output_path)
        )
        logger.debug(
            "event=turn_tts_complete engine=edge_tts turn=%d output=%s bytes=%d",
            turn.turn_id,
            output_path.name,
            output_path.stat().st_size,
        )
        return output_path

    output_path = out_dir / f"turn_{turn.turn_id:03d}.wav"
    if engine == "kokoro":
        kokoro = config["tts"]["kokoro"]
        voice = kokoro["voice_map"][turn.speaker]
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
                speed=_rate_to_kokoro_speed(turn.rate),
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

    turbo = config["tts"]["chatterbox_turbo"]
    reference_audio = turbo.get("reference_audio_map", {}).get(turn.speaker)
    logger.debug(
        "event=turn_tts_start engine=chatterbox_turbo turn=%d speaker=%s reference_audio=%s",
        turn.turn_id,
        turn.speaker,
        reference_audio or "builtin",
    )
    model = _get_chatterbox_turbo(
        turbo["device"], turbo.get("model_dir"), turbo.get("variant", "turbo") == "nano"
    )
    kwargs = {
        "temperature": turbo["temperature"],
        "top_p": turbo["top_p"],
        "top_k": turbo["top_k"],
        "repetition_penalty": turbo["repetition_penalty"],
    }
    if reference_audio:
        kwargs["audio_prompt_path"] = reference_audio
    audio = model.generate(turn.text, **kwargs)
    sf.write(output_path, audio.squeeze(0).detach().cpu().numpy(), model.sr)
    logger.debug(
        "event=turn_tts_complete engine=chatterbox_turbo turn=%d output=%s bytes=%d",
        turn.turn_id,
        output_path.name,
        output_path.stat().st_size,
    )
    return output_path


async def synthesize_all_turns(
    turns: list[NormalizedTurn], out_dir: Path, config: dict[str, Any]
) -> dict[int, Path]:
    return {
        turn.turn_id: await synthesize_turn(turn, out_dir, config) for turn in turns
    }


def describe_engine(config: dict[str, Any]) -> dict[str, Any]:
    """Return an engine and voice configuration snapshot for metadata traceability."""
    engine = get_engine(config)
    if engine == "edge_tts":
        return {"engine": "edge_tts", "voices": config["speaker_voice_map"]}
    if engine == "kokoro":
        kokoro = config["tts"]["kokoro"]
        return {
            "engine": "kokoro",
            "model": "Kokoro-82M",
            "sample_rate": kokoro["sample_rate"],
            "voices": kokoro["voice_map"],
        }
    turbo = config["tts"]["chatterbox_turbo"]
    variant = turbo.get("variant", "turbo")
    return {
        "engine": "chatterbox_turbo",
        "model": f"chatterbox-{variant}",
        "device": turbo["device"],
        "reference_audio": turbo.get("reference_audio_map", {}),
        "generation_params": {
            "temperature": turbo["temperature"],
            "top_p": turbo["top_p"],
            "top_k": turbo["top_k"],
            "repetition_penalty": turbo["repetition_penalty"],
        },
    }
