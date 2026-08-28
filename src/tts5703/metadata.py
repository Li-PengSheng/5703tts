"""Stage 5: write the turn-label-audio alignment contract."""

import json
from pathlib import Path

from .assemble import TurnTiming


def build_metadata(
    dialogue_id: str,
    clean_path: Path,
    telephone_path: Path,
    timings: list[TurnTiming],
    turn_audio_paths: dict[int, Path],
    engine_info: dict,
) -> dict:
    return {
        "dialogue_id": dialogue_id,
        "clean_audio": clean_path.name,
        "telephone_audio": telephone_path.name,
        # Record the engine and configuration used for this render so datasets
        # remain traceable during future engine-comparison experiments.
        "tts": engine_info,
        "turns": [
            {
                "turn_id": timing.turn_id,
                "speaker": timing.speaker,
                "text": timing.text,
                "label": timing.label,
                "turn_audio": turn_audio_paths[timing.turn_id].name,
                "rate": timing.rate,
                "pause_before_ms": timing.pause_before_ms,
                "pause_after_ms": timing.pause_after_ms,
                "emotion": timing.emotion,
                "arousal": timing.arousal,
                "coarse_affect": timing.coarse_affect,
                "paralinguistic_events": timing.paralinguistic_events,
                "start_time": timing.start_sec,
                "end_time": timing.end_sec,
            }
            for timing in timings
        ],
    }


def write_metadata(metadata: dict, out_dir: Path) -> Path:
    output_path = out_dir / f"{metadata['dialogue_id']}_metadata.json"
    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_path
