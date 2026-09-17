"""Pure semantic identities for final-record batch resume."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .backend_info import final_controlled_tts_backend_identity
from .input_records import InputRecord, canonical_json_sha256

FINAL_BATCH_MANIFEST_VERSION = "2.0"


class BatchIdentityError(ValueError):
    """Raised when a sidecar entry cannot identify one final render."""


def _nonblank(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BatchIdentityError(f"{path} must be a nonblank string")
    return value.strip()


def _sha256(value: Any, path: str) -> str:
    value = _nonblank(value, path).lower()
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise BatchIdentityError(f"{path} must be exactly 64 hexadecimal characters")
    return value


def materialization_component(
    sidecar_entry: dict[str, Any],
    *,
    registry_sha256: str,
    active_speakers_sha256: str,
) -> dict[str, Any]:
    """Return only per-dialogue speaker semantics, never sidecar location."""
    roles = sidecar_entry.get("roles")
    if not isinstance(roles, dict) or set(roles) != {"caller", "counsellor"}:
        raise BatchIdentityError(
            "sidecar roles must contain exactly caller and counsellor"
        )
    normalized: dict[str, dict[str, str]] = {}
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
        reference = role.get("higgs_reference")
        if not isinstance(reference, dict):
            raise BatchIdentityError(
                f"sidecar.roles.{logical_role}.higgs_reference must be an object"
            )
        declared_reference = _nonblank(
            reference.get("reference_wav"),
            f"sidecar.roles.{logical_role}.higgs_reference.reference_wav",
        )
        normalized[logical_role] = {
            "upstream_role": upstream_role,
            "render_speaker_id": _nonblank(
                role.get("render_speaker_id"),
                f"sidecar.roles.{logical_role}.render_speaker_id",
            ),
            "portable_reference_wav": (
                None if Path(declared_reference).is_absolute() else declared_reference
            ),
            "reference_sha256": _sha256(
                reference.get("sha256"),
                f"sidecar.roles.{logical_role}.higgs_reference.sha256",
            ),
        }
    return {
        "roles": normalized,
        "registry_sha256": _sha256(registry_sha256, "sidecar.registry_sha256"),
        "active_speakers_sha256": _sha256(
            active_speakers_sha256, "sidecar.active_speakers_sha256"
        ),
    }


def final_backend_identity(
    config: dict[str, Any], materialization: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Reuse the Phase-4A identity helper without preparing or mapping a turn."""
    roles = materialization["roles"].values() if materialization else ()
    proxy = SimpleNamespace(
        turns=tuple(
            SimpleNamespace(
                render_speaker_id=role["render_speaker_id"],
                reference_wav=role["portable_reference_wav"],
                reference_sha256=role["reference_sha256"],
            )
            for role in roles
        )
    )
    return final_controlled_tts_backend_identity(config, proxy)


def render_fingerprint_components(
    record: InputRecord,
    *,
    config_sha256: str,
    config: dict[str, Any],
    sidecar_entry: dict[str, Any],
    registry_sha256: str,
    active_speakers_sha256: str,
) -> dict[str, Any]:
    """Build the explicit semantic component object hashed for final resume."""
    materialization = materialization_component(
        sidecar_entry,
        registry_sha256=registry_sha256,
        active_speakers_sha256=active_speakers_sha256,
    )
    return {
        "source": {
            "dialogue_id": record.dialogue_id,
            "record_sha256": record.record_sha256,
        },
        "configuration": {"config_sha256": _sha256(config_sha256, "config_sha256")},
        "controlled_tts_backend": final_backend_identity(config, materialization),
        "materialization": materialization,
        "exclusion_decision": {"excluded": False},
    }


def render_fingerprint(components: dict[str, Any]) -> str:
    return canonical_json_sha256(components)
