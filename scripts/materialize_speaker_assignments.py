"""Create a backend-aware speaker sidecar without rewriting source dialogues.

Inputs are the final dialogue source, assignment rows, speaker registry, and
active-speaker list. Output is a separate dialogue sidecar linking source roles
to production render speakers and the explicitly requested ``higgs``,
``cosyvoice``, or ``both`` reference contracts.

Higgs approval is independent of CosyVoice ``primary_reference``. A registry
prompt may be a Higgs review candidate, but it is never emitted as an approved
``higgs_reference`` unless that separate field explicitly says
``approval_status=production_approved`` and its live SHA matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__:
    from . import assign_dialogue_speakers as assign
else:
    import assign_dialogue_speakers as assign

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEAKER_POOL_DIR = PROJECT_ROOT / "data" / "speaker_pool" / "vctk_v0.1"
DEFAULT_ACTIVE_SPEAKERS = SPEAKER_POOL_DIR / "active_speakers.json"
DEFAULT_REGISTRY = SPEAKER_POOL_DIR / "speaker_registry.json"
COSYVOICE3_END_OF_PROMPT = "<|endofprompt|>"
COSYVOICE3_ZERO_SHOT_PROMPT_PREFIX = (
    f"You are a helpful assistant.{COSYVOICE3_END_OF_PROMPT}"
)


class SpeakerMaterializationError(ValueError):
    """Raised when assignment records cannot be resolved into renderer inputs."""


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SpeakerMaterializationError(f"{label} not found: {path}") from error
    except json.JSONDecodeError as error:
        raise SpeakerMaterializationError(
            f"{label} is not valid JSON ({path}): {error}"
        ) from error
    if not isinstance(value, dict):
        raise SpeakerMaterializationError(f"{label} must contain a JSON object: {path}")
    return value


def load_assignments(path: Path) -> dict[str, dict[str, Any]]:
    """Index assignment JSONL by dialogue_id. Duplicate records are an error."""
    if not path.is_file():
        raise SpeakerMaterializationError(f"Assignment file not found: {path}")
    indexed: dict[str, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise SpeakerMaterializationError(
                f"Assignment line {line_number} is not valid JSON ({path}): {error}"
            ) from error
        if not isinstance(record, dict):
            raise SpeakerMaterializationError(
                f"Assignment line {line_number} must be a JSON object: {path}"
            )
        dialogue_id = record.get("dialogue_id")
        if not isinstance(dialogue_id, str) or not dialogue_id.strip():
            raise SpeakerMaterializationError(
                f"Assignment line {line_number} is missing dialogue_id: {path}"
            )
        dialogue_id = dialogue_id.strip()
        if dialogue_id in indexed:
            raise SpeakerMaterializationError(
                f"Duplicate assignment record for dialogue_id {dialogue_id!r}"
            )
        roles = record.get("role_assignments")
        if not isinstance(roles, dict) or not roles:
            raise SpeakerMaterializationError(
                f"Assignment for {dialogue_id!r} has no role_assignments"
            )
        cleaned: dict[str, str] = {}
        for role, speaker_id in roles.items():
            if not isinstance(role, str) or not role.strip():
                raise SpeakerMaterializationError(
                    f"Assignment for {dialogue_id!r} has an invalid role name"
                )
            if not isinstance(speaker_id, str) or not speaker_id.strip():
                raise SpeakerMaterializationError(
                    f"Assignment for {dialogue_id!r} role {role!r} has an "
                    "invalid speaker_id"
                )
            cleaned[role.strip()] = speaker_id.strip()
        record = dict(record)
        record["dialogue_id"] = dialogue_id
        record["role_assignments"] = cleaned
        indexed[dialogue_id] = record
    if not indexed:
        raise SpeakerMaterializationError(
            f"Assignment file contains no records: {path}"
        )
    return indexed


def _registry_by_id(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    speakers = registry.get("speakers")
    if not isinstance(speakers, list) or not speakers:
        raise SpeakerMaterializationError("Speaker registry lists no speakers")
    by_id: dict[str, dict[str, Any]] = {}
    for position, entry in enumerate(speakers, start=1):
        if not isinstance(entry, dict):
            raise SpeakerMaterializationError(
                f"Speaker registry entry {position} must be a JSON object"
            )
        speaker_id = entry.get("speaker_id")
        if not isinstance(speaker_id, str) or not speaker_id.strip():
            raise SpeakerMaterializationError(
                f"Speaker registry entry {position} is missing speaker_id"
            )
        by_id[speaker_id.strip()] = entry
    return by_id


def format_cosyvoice3_zero_shot_prompt_text(raw_transcript: str) -> str:
    """Prefix the registry transcript the way CosyVoice3 zero-shot expects."""
    if not isinstance(raw_transcript, str) or not raw_transcript.strip():
        raise SpeakerMaterializationError(
            "Registry prompt_text must be a non-empty string"
        )
    if COSYVOICE3_END_OF_PROMPT in raw_transcript:
        raise SpeakerMaterializationError(
            "Registry prompt_text already contains "
            f"{COSYVOICE3_END_OF_PROMPT}; store the raw VCTK transcript only"
        )
    return f"{COSYVOICE3_ZERO_SHOT_PROMPT_PREFIX}{raw_transcript}"


def _resolve_reference(
    speaker_id: str,
    entry: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, str]:
    primary = entry.get("primary_reference")
    if not isinstance(primary, dict):
        raise SpeakerMaterializationError(
            f"Speaker {speaker_id} has no primary_reference in the registry"
        )
    wav_value = primary.get("prompt_wav")
    transcript = primary.get("prompt_text")
    if not isinstance(wav_value, str) or not wav_value.strip():
        raise SpeakerMaterializationError(
            f"Speaker {speaker_id} primary_reference is missing prompt_wav"
        )
    wav_rel = wav_value.strip()
    wav_path = Path(wav_rel)
    if not wav_path.is_absolute():
        wav_path = project_root / wav_rel
    if not wav_path.is_file():
        raise SpeakerMaterializationError(
            f"Reference WAV missing for {speaker_id}: {wav_path}"
        )
    expected_hash = primary.get("sha256")
    if isinstance(expected_hash, str) and expected_hash.strip():
        actual_hash = _sha256_file(wav_path)
        if actual_hash != expected_hash.strip():
            raise SpeakerMaterializationError(
                f"SHA-256 mismatch for {speaker_id} reference {wav_path}: "
                f"registry={expected_hash.strip()} file={actual_hash}"
            )
    formatted = format_cosyvoice3_zero_shot_prompt_text(
        transcript if isinstance(transcript, str) else ""
    )
    source_speaker = entry.get("source_speaker_id")
    return {
        "speaker_id": speaker_id,
        "source_speaker_id": (
            source_speaker.strip()
            if isinstance(source_speaker, str) and source_speaker.strip()
            else ""
        ),
        "prompt_wav": wav_rel,
        "prompt_text": formatted,
        "raw_prompt_text": transcript.strip() if isinstance(transcript, str) else "",
        "sha256": (
            expected_hash.strip()
            if isinstance(expected_hash, str) and expected_hash.strip()
            else _sha256_file(wav_path)
        ),
    }


def _resolve_higgs_reference(
    speaker_id: str,
    entry: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, str]:
    reference = entry.get("higgs_reference")
    if not isinstance(reference, dict):
        raise SpeakerMaterializationError(
            f"Speaker {speaker_id} has no higgs_reference in the registry"
        )
    approval_status = reference.get("approval_status")
    if approval_status != "production_approved":
        raise SpeakerMaterializationError(
            f"Speaker {speaker_id} Higgs reference is not production approved: "
            f"approval_status={approval_status!r}"
        )
    wav_value = reference.get("reference_wav")
    if not isinstance(wav_value, str) or not wav_value.strip():
        raise SpeakerMaterializationError(
            f"Speaker {speaker_id} higgs_reference.reference_wav must be a "
            "non-empty string"
        )
    expected_hash = reference.get("sha256")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in expected_hash)
    ):
        raise SpeakerMaterializationError(
            f"Speaker {speaker_id} higgs_reference.sha256 must be exactly "
            "64 hexadecimal characters"
        )
    wav_rel = wav_value.strip()
    wav_path = Path(wav_rel)
    if not wav_path.is_absolute():
        wav_path = project_root / wav_path
    if not wav_path.is_file():
        raise SpeakerMaterializationError(
            f"Approved Higgs reference WAV missing for {speaker_id}: {wav_path}"
        )
    actual_hash = _sha256_file(wav_path)
    if actual_hash != expected_hash.lower():
        raise SpeakerMaterializationError(
            f"SHA-256 mismatch for {speaker_id} Higgs reference {wav_path}: "
            f"registry={expected_hash.lower()} file={actual_hash}"
        )
    return {"reference_wav": wav_rel, "sha256": actual_hash}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def materialize_speaker_assignments(
    *,
    input_path: Path,
    assignments_path: Path,
    registry_path: Path = DEFAULT_REGISTRY,
    active_speakers_path: Path = DEFAULT_ACTIVE_SPEAKERS,
    manifest_path: Path,
    project_root: Path = PROJECT_ROOT,
    backend: str,
) -> dict[str, Any]:
    """Validate all four inputs and emit the selected backend sidecar contract.

    The returned/written manifest records ``source_records_rewritten=False``;
    assignments and reference provenance remain external to final source JSON.
    """
    if backend not in {"higgs", "cosyvoice", "both"}:
        raise SpeakerMaterializationError(
            "backend must be one of: higgs, cosyvoice, both"
        )
    needs_cosyvoice = backend in {"cosyvoice", "both"}
    needs_higgs = backend in {"higgs", "both"}
    try:
        pool = assign.load_and_validate_active_pool(active_speakers_path, registry_path)
        dialogues = assign.load_dialogues(input_path)
        registry = assign.load_registry(registry_path)
    except assign.SpeakerAssignmentError as error:
        raise SpeakerMaterializationError(str(error)) from error

    assignments = load_assignments(assignments_path)
    registry_entries = _registry_by_id(registry)
    active_ids = set(pool.active_speaker_ids)
    excluded_ids = {item.speaker_id: item.reason for item in pool.exclusions}
    dialogue_ids = {dialogue.dialogue_id for dialogue in dialogues}
    extra_assignments = sorted(set(assignments) - dialogue_ids)
    if extra_assignments:
        raise SpeakerMaterializationError(
            "Assignment file references unknown dialogue_id values: "
            + ", ".join(extra_assignments)
        )

    cosyvoice_references: dict[str, dict[str, str]] = {}
    higgs_references: dict[str, dict[str, str]] = {}
    used_speaker_ids: set[str] = set()
    dialogue_rows: list[dict[str, Any]] = []

    for dialogue in dialogues:
        record = assignments.get(dialogue.dialogue_id)
        if record is None:
            raise SpeakerMaterializationError(
                f"No assignment record for dialogue_id {dialogue.dialogue_id!r}"
            )
        role_assignments = record["role_assignments"]
        needed = set(dialogue.roles)
        have = set(role_assignments)
        missing = sorted(needed - have)
        extra_roles = sorted(have - needed)
        if missing:
            raise SpeakerMaterializationError(
                f"Dialogue {dialogue.dialogue_id!r} missing role assignment for: "
                + ", ".join(missing)
            )
        if extra_roles:
            raise SpeakerMaterializationError(
                f"Assignment for {dialogue.dialogue_id!r} has extra role(s): "
                + ", ".join(extra_roles)
            )
        if len(set(role_assignments.values())) != len(role_assignments):
            raise SpeakerMaterializationError(
                f"Assignment for {dialogue.dialogue_id!r} maps distinct roles "
                "to the same speaker_id"
            )

        for role, speaker_id in role_assignments.items():
            if speaker_id in excluded_ids:
                raise SpeakerMaterializationError(
                    f"Speaker {speaker_id} assigned to {dialogue.dialogue_id!r} "
                    f"role {role!r} is excluded from the active pool "
                    f"({excluded_ids[speaker_id]})"
                )
            if speaker_id not in registry_entries:
                raise SpeakerMaterializationError(
                    f"Unknown speaker_id {speaker_id!r} assigned to "
                    f"{dialogue.dialogue_id!r} role {role!r}"
                )
            if speaker_id not in active_ids:
                raise SpeakerMaterializationError(
                    f"Speaker {speaker_id} assigned to {dialogue.dialogue_id!r} "
                    f"role {role!r} is not in the active speaker pool"
                )
            used_speaker_ids.add(speaker_id)
            if needs_cosyvoice and speaker_id not in cosyvoice_references:
                cosyvoice_references[speaker_id] = _resolve_reference(
                    speaker_id,
                    registry_entries[speaker_id],
                    project_root=project_root,
                )
            if needs_higgs and speaker_id not in higgs_references:
                higgs_references[speaker_id] = _resolve_higgs_reference(
                    speaker_id,
                    registry_entries[speaker_id],
                    project_root=project_root,
                )

        scenario_speakers = dialogue.raw["scenario"]["speakers"]
        roles: dict[str, Any] = {}
        for role, speaker_id in role_assignments.items():
            role_row: dict[str, Any] = {
                "upstream_role": dialogue.upstream_roles_by_logical[role],
                "upstream_scenario_speaker_id": scenario_speakers[role]["speaker_id"],
                "render_speaker_id": speaker_id,
            }
            if needs_cosyvoice:
                role_row["cosyvoice_reference"] = {
                    "prompt_wav": cosyvoice_references[speaker_id]["prompt_wav"],
                    "prompt_text": cosyvoice_references[speaker_id]["prompt_text"],
                    "sha256": cosyvoice_references[speaker_id]["sha256"],
                }
            if needs_higgs:
                role_row["higgs_reference"] = {
                    "reference_wav": higgs_references[speaker_id]["reference_wav"],
                    "sha256": higgs_references[speaker_id]["sha256"],
                }
            roles[role] = role_row
        dialogue_rows.append(
            {
                "source": {
                    "container_path": str(dialogue.source_path),
                    "format": dialogue.source_format,
                    "line_number": dialogue.line_number,
                    "record_sha256": dialogue.record_sha256,
                },
                "dialogue_id": dialogue.dialogue_id,
                "roles": roles,
            }
        )

    speaker_manifest = {
        speaker_id: {
            **(
                {
                    "primary_reference": {
                        "prompt_wav": cosyvoice_references[speaker_id]["prompt_wav"],
                        "sha256": cosyvoice_references[speaker_id]["sha256"],
                    }
                }
                if speaker_id in cosyvoice_references
                else {}
            ),
            **(
                {"higgs_reference": dict(higgs_references[speaker_id])}
                if speaker_id in higgs_references
                else {}
            ),
        }
        for speaker_id in sorted(used_speaker_ids)
    }
    manifest = {
        "materialization_mode": "speaker_sidecar",
        "source_records_rewritten": False,
        "assignment_file": str(assignments_path),
        "assignment_sha256": _sha256_file(assignments_path),
        "registry_path": str(registry_path),
        "registry_version": registry.get("speaker_pool_version"),
        "registry_sha256": _sha256_file(registry_path),
        "active_speakers_path": str(active_speakers_path),
        "active_speakers_sha256": _sha256_file(active_speakers_path),
        "project_root": str(project_root),
        "input_count": len(dialogues),
        "backend_target": backend,
        "speakers": speaker_manifest,
        "dialogues": dialogue_rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize final speaker assignments into a sidecar."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Canonical dialogue JSON/JSONL file or directory",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        required=True,
        help="speaker_assignments.jsonl from assign_dialogue_speakers.py",
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--active-speakers", type=Path, default=DEFAULT_ACTIVE_SPEAKERS)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Production speaker-sidecar output path",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root used to resolve registry prompt_wav paths",
    )
    parser.add_argument(
        "--backend",
        choices=("higgs", "cosyvoice", "both"),
        required=True,
        help="Reference type(s) to require and emit in the speaker sidecar",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = materialize_speaker_assignments(
            input_path=args.input,
            assignments_path=args.assignments,
            registry_path=args.registry,
            active_speakers_path=args.active_speakers,
            manifest_path=args.manifest,
            project_root=args.project_root,
            backend=args.backend,
        )
    except (SpeakerMaterializationError, OSError) as error:
        print(
            f"Speaker materialization failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
    print(f"Dialogues: {manifest['input_count']}")
    print(f"Speakers: {len(manifest['speakers'])}")
    print(f"Manifest: {manifest['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
