"""Trust boundary for the final Controlled-TTS nested dialogue contract.

This module validates the one accepted production shape; it does not adapt or
reinterpret legacy schemas.  Upstream ``User``/``Listener`` roles normalize to
logical ``caller``/``counsellor`` roles, while scenario ``speaker_id`` values
remain source identities rather than production render speaker IDs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from tts5703.controlled_tts.schema import (
    UpstreamValidationError,
    load_contract,
    normalize_turn,
)


class FinalInputValidationError(ValueError):
    """Raised when a final nested dialogue is structurally invalid."""


@dataclass(frozen=True)
class SpeakerIdentity:
    """Linked upstream role, logical role, and scenario speaker identity."""

    upstream_role: str
    logical_role: str
    upstream_scenario_speaker_id: str


@dataclass(frozen=True)
class ValidatedDialogue:
    """Validated final record plus the source-side speaker identity mapping."""

    dialogue_id: str
    raw: dict[str, Any]
    speaker_identities: dict[str, SpeakerIdentity]


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FinalInputValidationError(f"{path} must be a JSON object")
    return value


def _nonblank(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinalInputValidationError(f"{path} must be a nonblank string")
    return value


def validate_dialogue(raw: dict[str, Any]) -> ValidatedDialogue:
    """Validate only the final nested schema and preserve a source snapshot.

    Every turn must have ``turn_id`` and ``labels`` plus all six required
    controls: rate, arousal, affect, pause_before, pause_within, and hesitations.
    Best-effort fields are retained as requests; validation never promises that
    a selected backend can realize them.
    """
    raw = _mapping(raw, "dialogue")
    dialogue_id = _nonblank(raw.get("dialogue_id"), "dialogue_id")
    scenario = _mapping(raw.get("scenario"), "scenario")
    speakers = _mapping(scenario.get("speakers"), "scenario.speakers")
    identities: dict[str, SpeakerIdentity] = {}
    for upstream_role, logical_role in load_contract()["upstream_normalization"][
        "roles"
    ].items():
        role = _mapping(speakers.get(logical_role), f"scenario.speakers.{logical_role}")
        speaker_id = _nonblank(
            role.get("speaker_id"), f"scenario.speakers.{logical_role}.speaker_id"
        )
        identities[logical_role] = SpeakerIdentity(
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

    return ValidatedDialogue(
        dialogue_id=dialogue_id,
        raw=deepcopy(raw),
        speaker_identities=identities,
    )
