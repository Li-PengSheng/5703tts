"""Input validation and normalization; the sole W2-format boundary."""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "dialogue_schema.json"


class ValidationError(Exception):
    """Raised when dialogue input violates the pipeline contract."""


@lru_cache(maxsize=1)
def _load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@dataclass
class NormalizedTurn:
    turn_id: int
    speaker: str
    text: str
    label: str
    rate: str
    pause_after_ms: int
    emotion: str | None = None
    arousal: str | None = None
    paralinguistic_events: list[Any] = field(default_factory=list)


@dataclass
class NormalizedDialogue:
    dialogue_id: str
    turns: list[NormalizedTurn]


def validate_and_normalize(
    raw: dict[str, Any], config: dict[str, Any]
) -> NormalizedDialogue:
    try:
        jsonschema.validate(instance=raw, schema=_load_schema())
    except jsonschema.exceptions.ValidationError as error:
        raise ValidationError(
            f"Does not conform to schema ({error.json_path}): {error.message}"
        ) from error

    try:
        engine = config["tts"].get("engine", "edge_tts")
        if engine == "edge_tts":
            voice_map = config["speaker_voice_map"]
        elif engine == "kokoro":
            voice_map = config["tts"]["kokoro"]["voice_map"]
        elif engine == "chatterbox_turbo":
            # Turbo can use its bundled voice, so speaker availability still follows
            # the project-level speaker map; reference audio is optional per speaker.
            voice_map = config["speaker_voice_map"]
        else:
            raise ValidationError(
                f"Unsupported tts.engine: {engine} (available: edge_tts, kokoro, chatterbox_turbo)"
            )
        default_rate = config["tts"]["default_rate"]
        default_pause = config["pause"]["default_ms"]
    except KeyError as error:
        raise ValidationError(f"Configuration is missing required field: {error}") from error

    seen_ids: set[int] = set()
    turns: list[NormalizedTurn] = []
    for turn in raw["turns"]:
        turn_id = turn["turn_id"]
        if turn_id in seen_ids:
            raise ValidationError(f"Duplicate turn_id: {turn_id}")
        seen_ids.add(turn_id)
        speaker = turn["speaker"]
        if speaker not in voice_map:
            raise ValidationError(
                f"turn {turn_id}: speaker '{speaker}' has no configured voice; supported: {list(voice_map)}"
            )
        turns.append(
            NormalizedTurn(
                turn_id=turn_id,
                speaker=speaker,
                text=turn["text"],
                label=turn["label"],
                rate=turn.get("rate", default_rate),
                pause_after_ms=turn.get("pause_after_ms", default_pause),
                emotion=turn.get("emotion"),
                arousal=turn.get("arousal"),
                paralinguistic_events=turn.get("paralinguistic_events", []),
            )
        )

    ids = [turn.turn_id for turn in turns]
    if ids != sorted(ids):
        raise ValidationError(f"turn_id values are not in increasing order: {ids}")
    return NormalizedDialogue(dialogue_id=raw["dialogue_id"], turns=turns)


def load_and_validate(json_path: Path, config: dict[str, Any]) -> NormalizedDialogue:
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValidationError(f"Unable to parse JSON: {error}") from error
    return validate_and_normalize(raw, config)
