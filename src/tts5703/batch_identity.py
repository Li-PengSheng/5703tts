"""Pure semantic identities for final-record batch resume."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from .backend_info import final_controlled_tts_backend_identity
from .final_references import FinalReferenceError, selected_reference
from .input_records import InputRecord, canonical_json_sha256

FINAL_BATCH_MANIFEST_VERSION = "2.0"


class BatchIdentityError(ValueError):
    """Raised when a sidecar entry cannot identify one final render."""


def _nonblank(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BatchIdentityError(f"{path} must be a nonblank string")
    return value.strip()


def materialization_component(
    sidecar_entry: dict[str, Any],
    *,
    engine: str = "higgs",
) -> dict[str, Any]:
    """Return only per-dialogue speaker semantics, never sidecar location."""
    roles = sidecar_entry.get("roles")
    if not isinstance(roles, dict) or set(roles) != {"caller", "counsellor"}:
        raise BatchIdentityError(
            "sidecar roles must contain exactly caller and counsellor"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for logical_role, expected_upstream in (
        ("caller", "User"),
        ("counsellor", "Listener"),
    ):
        role = roles[logical_role]
        if not isinstance(role, dict):
            raise BatchIdentityError(f"sidecar.roles.{logical_role} must be an object")
        upstream_role = _nonblank(
            role.get("upstream_role"), f"sidecar.roles.{logical_role}.upstream_role"
        )
        if upstream_role != expected_upstream:
            raise BatchIdentityError(
                f"sidecar.roles.{logical_role}.upstream_role must be "
                f"{expected_upstream!r}"
            )
        reference_key = f"{engine}_reference"
        try:
            reference = selected_reference(role, engine, logical_role)
        except FinalReferenceError as error:
            raise BatchIdentityError(str(error)) from error
        reference_identity = {
            "upstream_role": upstream_role,
            "render_speaker_id": _nonblank(
                role.get("render_speaker_id"),
                f"sidecar.roles.{logical_role}.render_speaker_id",
            ),
            reference_key: reference.identity(),
        }
        normalized[logical_role] = reference_identity
    return {"roles": normalized}


def final_backend_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Reuse the Phase-4A identity helper without preparing or mapping a turn."""
    proxy = SimpleNamespace(turns=())
    return final_controlled_tts_backend_identity(config, proxy)


def render_affecting_config(config: dict[str, Any]) -> dict[str, Any]:
    """Snapshot only shared final audio settings; backend identity is separate."""
    return {
        "engine": config["tts"]["engine"],
        "fade_ms": config.get("fade_ms", 5),
        "telephone": deepcopy(config.get("telephone")),
    }


def render_fingerprint_components(
    record: InputRecord,
    *,
    config: dict[str, Any],
    sidecar_entry: dict[str, Any],
) -> dict[str, Any]:
    """Build the explicit semantic component object hashed for final resume."""
    materialization = materialization_component(
        sidecar_entry,
        engine=config["tts"]["engine"],
    )
    return {
        "source": {
            "dialogue_id": record.dialogue_id,
            "record_sha256": record.record_sha256,
        },
        "configuration": render_affecting_config(config),
        "controlled_tts_backend": final_backend_identity(config),
        "materialization": materialization,
        "exclusion_decision": {"excluded": False},
    }


def render_fingerprint(components: dict[str, Any]) -> str:
    return canonical_json_sha256(components)
