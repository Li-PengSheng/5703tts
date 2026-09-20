"""Parse only the selected backend's speaker-reference contract.

Higgs and CosyVoice references are intentionally distinct: Higgs consumes
``higgs_reference.reference_wav``; CosyVoice consumes
``cosyvoice_reference.prompt_wav`` plus ``prompt_text``.  A CosyVoice primary
reference is not automatically an approved Higgs reference.

The declared path is retained for provenance and semantic identity.  The
resolved path is separately used for runtime access.  With ``verify_file=True``
the current file bytes must match the sidecar SHA before production use.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class FinalReferenceError(ValueError):
    """Raised when a selected final reference is malformed or stale."""


@dataclass(frozen=True)
class SelectedReference:
    """Reference for the selected backend only, with provenance and live path."""

    engine: str
    declared_path: str
    resolved_path: Path | None
    sha256: str
    prompt_text: str | None = None

    @property
    def path_field(self) -> str:
        return "reference_wav" if self.engine == "higgs" else "prompt_wav"

    def identity(self) -> dict[str, str]:
        """Return persisted semantic fields, excluding the machine-local path."""
        identity = {self.path_field: self.declared_path, "sha256": self.sha256}
        if self.prompt_text is not None:
            identity["prompt_text"] = self.prompt_text
        return identity


def _nonblank(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinalReferenceError(f"{path} must be a nonblank string")
    return value.strip()


def _sha256(value: Any, path: str) -> str:
    digest = _nonblank(value, path).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise FinalReferenceError(f"{path} must be exactly 64 hexadecimal characters")
    return digest


def selected_reference(
    role: Any,
    engine: str,
    logical_role: str,
    *,
    project_root: Path | None = None,
    verify_file: bool = False,
) -> SelectedReference:
    """Parse the selected contract and optionally verify its current bytes.

    Unselected backend reference fields are deliberately ignored, so their
    absence or changes cannot affect a run using the other backend.
    """
    if engine == "higgs":
        reference_key, path_field = "higgs_reference", "reference_wav"
    elif engine == "cosyvoice":
        reference_key, path_field = "cosyvoice_reference", "prompt_wav"
    else:
        raise FinalReferenceError(f"Unsupported final reference engine: {engine!r}")
    if not isinstance(role, dict):
        raise FinalReferenceError(f"sidecar.roles.{logical_role} must be an object")
    reference = role.get(reference_key)
    if not isinstance(reference, dict):
        raise FinalReferenceError(
            f"sidecar.roles.{logical_role}.{reference_key} must be an object"
        )
    prefix = f"sidecar.roles.{logical_role}.{reference_key}"
    declared = _nonblank(reference.get(path_field), f"{prefix}.{path_field}")
    digest = _sha256(reference.get("sha256"), f"{prefix}.sha256")
    prompt_text = None
    if engine == "cosyvoice":
        value = reference.get("prompt_text")
        if not isinstance(value, str) or not value.strip():
            raise FinalReferenceError(f"{prefix}.prompt_text must be a nonblank string")
        prompt_text = value

    resolved = None
    if project_root is not None:
        resolved = Path(declared)
        if not resolved.is_absolute():
            resolved = project_root / resolved
        resolved = resolved.resolve()
    if verify_file:
        if resolved is None:
            raise FinalReferenceError("project_root is required to verify a reference")
        if not resolved.is_file():
            raise FinalReferenceError(
                f"Selected {reference_key} file is missing: {resolved}"
            )
        try:
            with resolved.open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
        except OSError as error:
            raise FinalReferenceError(
                f"Selected {reference_key} file is unreadable: {resolved}"
            ) from error
        if actual != digest:
            raise FinalReferenceError(
                f"Selected {reference_key} SHA-256 mismatch: {resolved}"
            )
    return SelectedReference(engine, declared, resolved, digest, prompt_text)
