"""Model-independent input validation and normalization boundary."""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

from .config import VALID_ENGINES

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "dialogue_schema.json"

# Temporary compatibility schema for the versionless, flat input format. Keep this
# separate from the canonical schema so legacy percentages cannot be mistaken for
# v0.2 semantic rates. Remove this schema and its normalizer once legacy inputs have
# been migrated.
_LEGACY_DIALOGUE_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "required": ["dialogue_id", "turns"],
    "properties": {
        "dialogue_id": {"type": "string", "minLength": 1},
        "turns": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["turn_id", "speaker", "text", "label"],
                "properties": {
                    "turn_id": {"type": "integer"},
                    "speaker": {"type": "string", "minLength": 1},
                    "text": {"type": "string", "minLength": 1},
                    "label": {
                        "type": "string",
                        "enum": ["normal", "alert", "confirm"],
                    },
                    "rate": {"type": "string", "pattern": r"^[+-][0-9]+%$"},
                    "pause_after_ms": {"type": "integer", "minimum": 0},
                    "emotion": {"type": ["string", "null"]},
                    "arousal": {"type": ["string", "null"]},
                    "paralinguistic_events": {"type": "array"},
                },
            },
        },
    },
}


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
    pause_before_ms: int
    pause_after_ms: int
    arousal: str | None = None
    coarse_affect: str | None = None
    emotion: str | None = None
    paralinguistic_events: list[Any] = field(default_factory=list)


@dataclass
class NormalizedDialogue:
    dialogue_id: str
    turns: list[NormalizedTurn]


def _validate_schema(raw: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        jsonschema.validate(instance=raw, schema=schema)
    except jsonschema.exceptions.ValidationError as error:
        raise ValidationError(
            f"Does not conform to schema ({error.json_path}): {error.message}"
        ) from error


def _normalize_v0_2_turn(
    turn: dict[str, Any], default_pause_after_ms: int
) -> NormalizedTurn:
    """Normalize canonical semantic intent without backend-specific conversion."""
    acoustic_spec = turn["acoustic_spec"]
    return NormalizedTurn(
        turn_id=turn["turn_id"],
        speaker=turn["speaker"],
        text=turn["text"],
        label=turn["label"],
        rate=acoustic_spec.get("rate", "normal"),
        pause_before_ms=acoustic_spec.get("pause_before_ms", 0),
        pause_after_ms=acoustic_spec.get("pause_after_ms", default_pause_after_ms),
        arousal=acoustic_spec.get("arousal"),
        coarse_affect=acoustic_spec.get("coarse_affect"),
        emotion=acoustic_spec.get("emotion"),
        paralinguistic_events=list(acoustic_spec.get("paralinguistic_events", [])),
    )


def _normalize_legacy_turn(
    turn: dict[str, Any], default_rate: str, default_pause_after_ms: int
) -> NormalizedTurn:
    """Normalize temporary flat input while preserving its percentage rate."""
    return NormalizedTurn(
        turn_id=turn["turn_id"],
        speaker=turn["speaker"],
        text=turn["text"],
        label=turn["label"],
        rate=turn.get("rate", default_rate),
        pause_before_ms=0,
        pause_after_ms=turn.get("pause_after_ms", default_pause_after_ms),
        arousal=turn.get("arousal"),
        coarse_affect=None,
        emotion=turn.get("emotion"),
        paralinguistic_events=list(turn.get("paralinguistic_events", [])),
    )


def validate_and_normalize(
    raw: dict[str, Any], config: dict[str, Any]
) -> NormalizedDialogue:
    is_v0_2 = isinstance(raw, dict) and "schema_version" in raw
    _validate_schema(raw, _load_schema() if is_v0_2 else _LEGACY_DIALOGUE_SCHEMA)

    try:
        engine = config["tts"].get("engine", "edge_tts")
        if engine == "edge_tts":
            voice_map = config["speaker_voice_map"]
        elif engine == "kokoro":
            voice_map = config["tts"]["kokoro"]["voice_map"]
        elif engine == "cosyvoice":
            voice_map = config["tts"]["cosyvoice"]["voice_map"]
        elif engine == "chatterbox_turbo":
            # Turbo can use its bundled voice, so speaker availability still follows
            # the project-level speaker map; reference audio is optional per speaker.
            voice_map = config["speaker_voice_map"]
        else:
            raise ValidationError(
                f"Unsupported tts.engine: {engine} "
                f"(available: {', '.join(sorted(VALID_ENGINES))})"
            )
        default_rate = config["tts"]["default_rate"]
        default_pause = config["pause"]["default_ms"]
    except KeyError as error:
        raise ValidationError(
            f"Configuration is missing required field: {error}"
        ) from error

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
            _normalize_v0_2_turn(turn, default_pause)
            if is_v0_2
            else _normalize_legacy_turn(turn, default_rate, default_pause)
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
