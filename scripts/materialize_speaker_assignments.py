"""Resolve assignment JSONL into render-ready dialogues and CosyVoice voice_map.

Canonical dialogues still use role names (``caller``, ``counsellor``, ...).
Assignment maps those roles to persistent ``speaker_id`` values. This helper
rewrites each turn's ``speaker`` field to that ID and builds a CosyVoice
``voice_map`` keyed by ``speaker_id``, so the same role name can map to
different speakers in different dialogues without a global caller/counsellor
voice.

It does not assign speakers, synthesise audio, or change the renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import yaml

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


def _load_assign_module() -> Any:
    path = Path(__file__).resolve().with_name("assign_dialogue_speakers.py")
    spec = importlib.util.spec_from_file_location("assign_dialogue_speakers", path)
    if spec is None or spec.loader is None:
        raise SpeakerMaterializationError(f"Cannot load assignment helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


assign = _load_assign_module()


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


def _require_empty_directory(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        return
    if not path.is_dir():
        raise SpeakerMaterializationError(
            f"Render output path is not a directory: {path}"
        )
    if any(path.iterdir()):
        raise SpeakerMaterializationError(
            f"Render output directory is not empty: {path}. "
            "Use a new or empty directory so stale JSON cannot mix with this batch."
        )


def _rewrite_dialogue(
    raw: dict[str, Any], role_assignments: dict[str, str]
) -> dict[str, Any]:
    turns = raw.get("turns")
    if not isinstance(turns, list):
        raise SpeakerMaterializationError(
            f"Dialogue {raw.get('dialogue_id')!r} has no turns array"
        )
    rewritten = json.loads(json.dumps(raw))
    for turn in rewritten["turns"]:
        role = turn["speaker"]
        turn["speaker"] = role_assignments[role]
    return rewritten


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=2**20,
        ),
        encoding="utf-8",
    )


def materialize_speaker_assignments(
    *,
    input_path: Path,
    assignments_path: Path,
    registry_path: Path = DEFAULT_REGISTRY,
    active_speakers_path: Path = DEFAULT_ACTIVE_SPEAKERS,
    output_dir: Path,
    config_out: Path,
    base_config_path: Path,
    manifest_path: Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Rewrite role names to speaker IDs and emit a CosyVoice speaker voice_map."""
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

    used_speakers: dict[str, dict[str, str]] = {}
    dialogue_rows: list[dict[str, Any]] = []
    output_names: dict[str, str] = {}

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
            if speaker_id not in used_speakers:
                used_speakers[speaker_id] = _resolve_reference(
                    speaker_id,
                    registry_entries[speaker_id],
                    project_root=project_root,
                )
        output_name = dialogue.source_path.name
        previous = output_names.get(output_name)
        if previous is not None:
            raise SpeakerMaterializationError(
                f"Two inputs would write the same render filename {output_name!r}: "
                f"{previous} and {dialogue.source_path}"
            )
        output_names[output_name] = str(dialogue.source_path)
        dialogue_rows.append(
            {
                "dialogue_id": dialogue.dialogue_id,
                "source_input_path": str(dialogue.source_path),
                "materialized_input_path": str(output_dir / output_name),
                "roles": list(dialogue.roles),
                "role_assignments": dict(role_assignments),
                "speakers": {
                    role: {
                        "speaker_id": speaker_id,
                        "source_speaker_id": used_speakers[speaker_id][
                            "source_speaker_id"
                        ],
                        "prompt_wav": used_speakers[speaker_id]["prompt_wav"],
                        "raw_prompt_text": used_speakers[speaker_id]["raw_prompt_text"],
                    }
                    for role, speaker_id in role_assignments.items()
                },
            }
        )

    try:
        base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SpeakerMaterializationError(
            f"Base config not found: {base_config_path}"
        ) from error
    except yaml.YAMLError as error:
        raise SpeakerMaterializationError(
            f"Base config is not valid YAML ({base_config_path}): {error}"
        ) from error
    if not isinstance(base_config, dict):
        raise SpeakerMaterializationError(
            f"Base config must contain a YAML mapping: {base_config_path}"
        )
    try:
        cosyvoice = base_config["tts"]["cosyvoice"]
    except (KeyError, TypeError) as error:
        raise SpeakerMaterializationError(
            "Base config must contain tts.cosyvoice so speaker references can "
            "be materialized without inventing renderer settings"
        ) from error
    if not isinstance(cosyvoice, dict):
        raise SpeakerMaterializationError("tts.cosyvoice must be a mapping")

    voice_map = {
        speaker_id: {
            "prompt_wav": used_speakers[speaker_id]["prompt_wav"],
            "prompt_text": used_speakers[speaker_id]["prompt_text"],
        }
        for speaker_id in sorted(used_speakers)
    }
    cosyvoice["voice_map"] = voice_map

    if manifest_path is None:
        manifest_path = config_out.parent / "materialization_manifest.json"

    _require_empty_directory(output_dir)
    config_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    for dialogue, row in zip(dialogues, dialogue_rows, strict=True):
        raw = _load_json_object(dialogue.source_path, "Dialogue")
        rewritten = _rewrite_dialogue(raw, row["role_assignments"])
        dest = output_dir / dialogue.source_path.name
        _write_json(dest, rewritten)
        row["materialized_input_path"] = str(dest)

    _write_yaml(config_out, base_config)
    manifest = {
        "assignment_file": str(assignments_path),
        "assignment_sha256": _sha256_file(assignments_path),
        "registry_path": str(registry_path),
        "registry_version": registry.get("speaker_pool_version"),
        "active_speakers_path": str(active_speakers_path),
        "base_config": str(base_config_path),
        "project_root": str(project_root),
        "input_count": len(dialogues),
        "output_dir": str(output_dir),
        "config_out": str(config_out),
        "speakers": {
            speaker_id: {
                "source_speaker_id": used_speakers[speaker_id]["source_speaker_id"],
                "prompt_wav": used_speakers[speaker_id]["prompt_wav"],
                "sha256": used_speakers[speaker_id]["sha256"],
            }
            for speaker_id in sorted(used_speakers)
        },
        "dialogues": dialogue_rows,
    }
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize speaker assignments into render-ready dialogues and "
            "a CosyVoice speaker voice_map. Does not run TTS."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Canonical dialogue JSON file or directory (role names in speaker)",
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
        "--output",
        type=Path,
        required=True,
        help="Empty directory for speaker-resolved render JSON",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        required=True,
        help="Path to write the materialized CosyVoice config",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        required=True,
        help="Existing renderer config whose unrelated fields are preserved",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Materialization manifest path (default: next to --config-out)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root used to resolve registry prompt_wav paths",
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
            output_dir=args.output,
            config_out=args.config_out,
            base_config_path=args.base_config,
            manifest_path=args.manifest,
            project_root=args.project_root,
        )
    except (SpeakerMaterializationError, OSError) as error:
        print(
            f"Speaker materialization failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
    print(f"Dialogues: {manifest['input_count']}")
    print(f"Speakers: {len(manifest['speakers'])}")
    print(f"Render input: {manifest['output_dir']}")
    print(f"Config: {manifest['config_out']}")
    print(f"Manifest: {manifest['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
