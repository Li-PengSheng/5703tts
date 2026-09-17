"""Schema-family detection and validation for non-rendered final inputs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any

from tts5703.controlled_tts.schema import (
    UpstreamValidationError,
    load_contract,
    normalize_turn,
)


class SchemaFamily(str, Enum):
    LEGACY = "legacy"
    V0_2 = "v0.2"
    FINAL_NESTED = "final_nested"


class SchemaFamilyError(ValueError):
    """Raised when a record is structurally ambiguous or unsupported."""


class FinalInputValidationError(ValueError):
    """Raised when a final nested dialogue is structurally invalid."""


@dataclass(frozen=True)
class FinalSpeakerIdentity:
    upstream_role: str
    logical_role: str
    upstream_scenario_speaker_id: str


@dataclass(frozen=True)
class FinalDialogue:
    dialogue_id: str
    raw: dict[str, Any]
    speaker_identities: dict[str, FinalSpeakerIdentity]


def detect_schema_family(raw: dict[str, Any]) -> SchemaFamily:
    """Classify by acoustic structure, never by schema_version alone."""
    if not isinstance(raw, dict):
        raise SchemaFamilyError("Dialogue record must be a JSON object")
    turns = raw.get("turns")
    turn_objects = turns if isinstance(turns, list) else []
    has_v02 = False
    has_final = False
    for turn in turn_objects:
        if not isinstance(turn, dict):
            continue
        turn_v02 = "acoustic_spec" in turn
        acoustic = turn.get("acoustic")
        turn_final = isinstance(acoustic, dict) and "required" in acoustic
        if turn_v02 and turn_final:
            raise SchemaFamilyError(
                "Hybrid record contains both acoustic_spec and acoustic.required"
            )
        has_v02 = has_v02 or turn_v02
        has_final = has_final or turn_final

    if has_v02 and has_final:
        raise SchemaFamilyError(
            "Hybrid record mixes acoustic_spec and acoustic.required turns"
        )

    version = raw.get("schema_version")
    if has_final:
        if version == "0.2":
            raise SchemaFamilyError(
                "Hybrid record declares schema_version 0.2 with acoustic.required"
            )
        return SchemaFamily.FINAL_NESTED
    if version == "0.2":
        if not isinstance(turns, list) or not turns:
            raise SchemaFamilyError(
                "schema_version 0.2 requires a non-empty turns list"
            )
        if not all(
            isinstance(turn, dict) and isinstance(turn.get("acoustic_spec"), dict)
            for turn in turns
        ):
            raise SchemaFamilyError(
                "schema_version 0.2 requires acoustic_spec on every turn"
            )
        return SchemaFamily.V0_2
    if has_v02:
        raise SchemaFamilyError("acoustic_spec records must declare schema_version 0.2")
    if version is not None:
        raise SchemaFamilyError(
            f"Unsupported schema_version without final nested structure: {version!r}"
        )
    return SchemaFamily.LEGACY


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FinalInputValidationError(f"{path} must be a JSON object")
    return value


def _nonblank(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinalInputValidationError(f"{path} must be a nonblank string")
    return value


def validate_final_dialogue(raw: dict[str, Any]) -> FinalDialogue:
    """Validate final structure while retaining the untouched nested record."""
    if detect_schema_family(raw) is not SchemaFamily.FINAL_NESTED:
        raise FinalInputValidationError("Dialogue is not a final nested record")
    dialogue_id = _nonblank(raw.get("dialogue_id"), "dialogue_id")
    scenario = _mapping(raw.get("scenario"), "scenario")
    speakers = _mapping(scenario.get("speakers"), "scenario.speakers")
    identities: dict[str, FinalSpeakerIdentity] = {}
    for upstream_role, logical_role in load_contract()["upstream_normalization"][
        "roles"
    ].items():
        role = _mapping(speakers.get(logical_role), f"scenario.speakers.{logical_role}")
        speaker_id = _nonblank(
            role.get("speaker_id"), f"scenario.speakers.{logical_role}.speaker_id"
        )
        identities[logical_role] = FinalSpeakerIdentity(
            upstream_role=upstream_role,
            logical_role=logical_role,
            upstream_scenario_speaker_id=speaker_id,
        )

    turns = raw.get("turns")
    if not isinstance(turns, list) or not turns:
        raise FinalInputValidationError(f"Dialogue {dialogue_id!r} has no turns")
    for position, value in enumerate(turns, start=1):
        turn = _mapping(value, f"turns[{position}]")
        if "turn_id" not in turn:
            raise FinalInputValidationError(f"turns[{position}].turn_id is required")
        if "labels" not in turn:
            raise FinalInputValidationError(f"turns[{position}].labels is required")
        try:
            normalize_turn(turn, dialogue_context=raw)
        except UpstreamValidationError as error:
            raise FinalInputValidationError(str(error)) from error

    return FinalDialogue(
        dialogue_id=dialogue_id,
        raw=deepcopy(raw),
        speaker_identities=identities,
    )
