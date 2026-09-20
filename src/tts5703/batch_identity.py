"""Pure semantic identity used by manifest-v2 resume decisions.

The render fingerprint covers canonical source identity, render-affecting
shared config, the selected backend's static identity, selected speaker/reference
materialization, and the exclusion decision. It intentionally excludes input
file location, JSON formatting, and unselected backend configuration.

Static backend identity records configured model/runtime paths and mapping
identity, but does not hash the full checkpoint or runtime environment. Live
artifact integrity and selected-reference bytes are checked separately during
resume; a matching fingerprint alone is never sufficient.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .backends.info import backend_static_identity as _backend_static_identity
from .input.records import InputRecord, canonical_json_sha256
from .speaker_references import FinalReferenceError, selected_reference

BATCH_MANIFEST_VERSION = "2.0"


class BatchIdentityError(ValueError):
    """Raised when a sidecar entry cannot identify one production render."""


def _nonblank(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BatchIdentityError(f"{path} must be a nonblank string")
    return value.strip()


def materialization_component(
    sidecar_entry: dict[str, Any],
    *,
    engine: str = "higgs",
) -> dict[str, Any]:
    """Return selected per-dialogue speaker semantics, never sidecar location."""
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


def backend_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Describe selected static backend semantics before dialogue preparation."""
    return _backend_static_identity(config)


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
    """Build the explicit semantic components hashed for resume.

    File/container location and formatting are absent because
    ``record_sha256`` already represents canonical source content. Only the
    selected backend and its selected references participate.
    """
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
        "controlled_tts_backend": backend_identity(config),
        "materialization": materialization,
        "exclusion_decision": {"excluded": False},
    }


def render_fingerprint(components: dict[str, Any]) -> str:
    """Hash the canonical semantic component object for manifest-v2 resume."""
    return canonical_json_sha256(components)
