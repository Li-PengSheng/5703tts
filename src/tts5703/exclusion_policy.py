"""Data-driven dialogue exclusion policies."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tts5703.input_records import canonical_json_sha256

EXCLUSION_POLICY_SCHEMA_VERSION = "1.0"


class ExclusionPolicyError(ValueError):
    """Raised when an exclusion policy is invalid."""


class _DuplicateKeyError(ValueError):
    pass


@dataclass(frozen=True)
class DialogueExclusion:
    dialogue_id: str
    reason: str
    provenance: Any = None


@dataclass(frozen=True, init=False)
class ExclusionPolicy:
    schema_version: str
    _exclusions: tuple[DialogueExclusion, ...] = field(repr=False)
    policy_sha256: str

    def __init__(
        self, schema_version: str, exclusions: tuple[DialogueExclusion, ...]
    ) -> None:
        snapshot = tuple(_copy_exclusion(item) for item in exclusions)
        semantic = {
            "schema_version": schema_version,
            "exclusions": [
                {
                    "dialogue_id": item.dialogue_id,
                    "reason": item.reason,
                    **(
                        {"provenance": item.provenance}
                        if item.provenance is not None
                        else {}
                    ),
                }
                for item in snapshot
            ],
        }
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "_exclusions", snapshot)
        object.__setattr__(self, "policy_sha256", canonical_json_sha256(semantic))

    @property
    def exclusions(self) -> tuple[DialogueExclusion, ...]:
        return tuple(_copy_exclusion(item) for item in self._exclusions)

    def exclusion_for(self, dialogue_id: str) -> DialogueExclusion | None:
        item = next(
            (item for item in self._exclusions if item.dialogue_id == dialogue_id), None
        )
        return _copy_exclusion(item) if item is not None else None


def _copy_exclusion(item: DialogueExclusion) -> DialogueExclusion:
    return DialogueExclusion(
        dialogue_id=item.dialogue_id,
        reason=item.reason,
        provenance=deepcopy(item.provenance),
    )


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def parse_exclusion_policy(raw: dict[str, Any]) -> ExclusionPolicy:
    if not isinstance(raw, dict):
        raise ExclusionPolicyError("Exclusion policy must be a JSON object")
    version = raw.get("schema_version")
    if version != EXCLUSION_POLICY_SCHEMA_VERSION:
        raise ExclusionPolicyError(
            f"schema_version must be {EXCLUSION_POLICY_SCHEMA_VERSION!r}"
        )
    values = raw.get("exclusions")
    if not isinstance(values, list):
        raise ExclusionPolicyError("exclusions must be a list")
    exclusions: list[DialogueExclusion] = []
    seen: set[str] = set()
    for position, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            raise ExclusionPolicyError(f"exclusions[{position}] must be an object")
        dialogue_id = value.get("dialogue_id")
        reason = value.get("reason")
        if not isinstance(dialogue_id, str) or not dialogue_id.strip():
            raise ExclusionPolicyError(
                f"exclusions[{position}].dialogue_id must be nonblank"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ExclusionPolicyError(
                f"exclusions[{position}].reason must be nonblank"
            )
        dialogue_id = dialogue_id.strip()
        if dialogue_id in seen:
            raise ExclusionPolicyError(f"Duplicate dialogue_id: {dialogue_id}")
        seen.add(dialogue_id)
        exclusions.append(
            DialogueExclusion(
                dialogue_id=dialogue_id,
                reason=reason.strip(),
                provenance=deepcopy(value.get("provenance")),
            )
        )
    exclusions.sort(key=lambda item: item.dialogue_id)
    try:
        return ExclusionPolicy(version, tuple(exclusions))
    except (TypeError, ValueError) as error:
        raise ExclusionPolicyError("provenance must contain JSON values") from error


def load_exclusion_policy(path: Path) -> ExclusionPolicy:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_non_json_constant,
        )
    except FileNotFoundError as error:
        raise ExclusionPolicyError(f"Exclusion policy not found: {path}") from error
    except ValueError as error:
        raise ExclusionPolicyError(f"Invalid exclusion policy JSON: {error}") from error
    return parse_exclusion_policy(raw)
