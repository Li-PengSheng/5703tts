"""Production batch orchestration, manifest construction, and resume."""

import argparse
import datetime
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any

from pydub import AudioSegment

from .batch_identity import (
    BATCH_MANIFEST_VERSION,
    BatchIdentityError,
    backend_identity,
    render_fingerprint,
    render_fingerprint_components,
)
from .config import VALID_ENGINES
from .input.contract import FinalInputValidationError, validate_dialogue
from .input.exclusions import ExclusionPolicy, load_exclusion_policy
from .input.records import InputRecord, InputRecordFailure
from .pipeline import run_dialogue
from .speaker_references import FinalReferenceError, selected_reference

_RENDERED = "rendered"
_RESUMED = "resumed"
_EXCLUDED = "excluded_known_issue"
_INPUT_ERROR = "input_error"
_RENDER_FAILED = "render_failed"
_RESUME_MANIFEST_HINT = "Repair or remove this file, or run without --resume to ignore previous batch state."
_MANAGED_TURN_ARTIFACT = re.compile(
    r"turn_(?:00[1-9]|0[1-9][0-9]|[1-9][0-9]{2,})"
    r"(?:\.wav|\.higgs_raw\.wav(?:\.part)?|\.higgs_processed\.part\.wav)"
)


def _write_batch_result(result: dict[str, Any], output_root: Path) -> Path:
    """Atomically write the authoritative result for this output root."""
    path = output_root / "batch_result.json"
    temporary_path = output_root / ".batch_result.json.tmp"
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)
    return path


def _load_previous_batch_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot resume from unreadable batch result {path}: {error}. "
            f"{_RESUME_MANIFEST_HINT}"
        ) from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("manifest_version") != BATCH_MANIFEST_VERSION
        or manifest.get("status") not in {"success", "partial_failure", "failure"}
        or not isinstance(manifest.get("results"), list)
    ):
        raise RuntimeError(
            f"Cannot resume from malformed batch result: {path}. {_RESUME_MANIFEST_HINT}"
        )
    return manifest


def _safe_dialogue_output_path(output_root: Path, dialogue_id: str) -> Path:
    """Return one direct child output path or reject unsafe dialogue IDs."""
    if (
        not isinstance(dialogue_id, str)
        or not dialogue_id.strip()
        or dialogue_id in {".", ".."}
        or "/" in dialogue_id
        or "\\" in dialogue_id
        or Path(dialogue_id).is_absolute()
    ):
        raise RuntimeError(f"Unsafe final dialogue_id for output path: {dialogue_id!r}")
    raw_candidate = output_root / dialogue_id
    if raw_candidate.is_symlink():
        raise RuntimeError(
            f"Final dialogue output path must not be a symlink: {dialogue_id!r}"
        )
    root = output_root.resolve()
    candidate = raw_candidate.resolve()
    if candidate.parent != root or not candidate.is_relative_to(root):
        raise RuntimeError(f"Final dialogue_id escapes output root: {dialogue_id!r}")
    return candidate


def _is_managed_dialogue_artifact(name: str, dialogue_id: str) -> bool:
    if name in {
        f"{dialogue_id}_clean.wav",
        f"{dialogue_id}_telephone.wav",
        f"{dialogue_id}_metadata.json",
    }:
        return True
    return _MANAGED_TURN_ARTIFACT.fullmatch(name) is not None


def _cleanup_managed_dialogue_artifacts(out_dir: Path, dialogue_id: str) -> None:
    """Remove only pipeline-owned direct-child files before a rerender."""
    if not out_dir.exists():
        return
    for path in out_dir.iterdir():
        if not _is_managed_dialogue_artifact(path.name, dialogue_id):
            continue
        if path.is_symlink() or path.is_file():
            path.unlink()
            continue
        raise RuntimeError(f"Managed artifact path is not a file: {path}")


class _DuplicateJsonKeyError(ValueError):
    pass


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _load_sidecar(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        sidecar = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Cannot load final speaker sidecar {path}: {error}"
        ) from error
    if not isinstance(sidecar, dict) or not isinstance(sidecar.get("dialogues"), list):
        raise TypeError("Final speaker sidecar must contain a dialogues list")
    indexed: dict[str, dict[str, Any]] = {}
    for entry in sidecar["dialogues"]:
        dialogue_id = entry.get("dialogue_id") if isinstance(entry, dict) else None
        if not isinstance(dialogue_id, str) or not dialogue_id.strip():
            raise RuntimeError("Final speaker sidecar has a blank dialogue_id")
        if dialogue_id in indexed:
            raise RuntimeError(
                f"Final speaker sidecar has duplicate dialogue_id {dialogue_id!r}"
            )
        indexed[dialogue_id] = entry
    return sidecar, indexed


def _index_previous_results(
    manifest: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    invalid: set[str] = set()
    for entry in manifest.get("results", []) if manifest else []:
        dialogue_id = entry.get("dialogue_id") if isinstance(entry, dict) else None
        if (
            not isinstance(dialogue_id, str)
            or not dialogue_id
            or dialogue_id in invalid
        ):
            continue
        if dialogue_id in indexed:
            indexed.pop(dialogue_id)
            invalid.add(dialogue_id)
        else:
            indexed[dialogue_id] = entry
    return indexed


def _resume_artifacts_valid(
    output_root: Path,
    input_record: InputRecord,
    *,
    engine: str,
    current_backend_identity: dict[str, Any],
) -> bool:
    """Validate persisted output structure before trusting a resume candidate."""
    dialogue_id = input_record.dialogue_id
    out_dir = _safe_dialogue_output_path(output_root, dialogue_id)
    expected_clean = f"{dialogue_id}_clean.wav"
    expected_telephone = f"{dialogue_id}_telephone.wav"
    try:
        metadata = json.loads(
            (out_dir / f"{dialogue_id}_metadata.json").read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
        if not isinstance(metadata, dict):
            return False
        current_implementation_id = current_backend_identity["control_mapping"][
            "implementation_id"
        ]
        tts = metadata.get("tts")
        if not isinstance(tts, dict):
            return False
        persisted_backend = tts.get("backend_identity")
        persisted_mapping = tts.get("control_mapping")
        if not isinstance(persisted_backend, dict) or not isinstance(
            persisted_mapping, dict
        ):
            return False
        backend_mapping = persisted_backend.get("control_mapping")
        if not isinstance(backend_mapping, dict):
            return False
        if (
            metadata.get("dialogue_id") != dialogue_id
            or metadata.get("clean_audio") != expected_clean
            or metadata.get("telephone_audio") != expected_telephone
            or metadata["provenance"]["record_sha256"] != input_record.record_sha256
            or tts.get("engine") != engine
            or persisted_backend.get("backend") != engine
            or persisted_mapping.get("implementation_id") != current_implementation_id
            or backend_mapping.get("implementation_id") != current_implementation_id
        ):
            return False

        source_turns = input_record.raw["turns"]
        metadata_turns = metadata["turns"]
        if not isinstance(metadata_turns, list) or len(metadata_turns) != len(
            source_turns
        ):
            return False

        previous_end = 0.0
        for ordinal, (source, persisted) in enumerate(
            zip(source_turns, metadata_turns, strict=True), start=1
        ):
            identity = persisted["source_identity"]
            execution = persisted["execution"]
            requested = persisted["requested"]["acoustic"]
            timing = persisted["timing"]
            expected_turn_audio = f"turn_{ordinal:03d}.wav"
            start_sec = timing["start_sec"]
            end_sec = timing["end_sec"]
            if (
                type(identity.get("ordinal")) is not int
                or identity.get("ordinal") != ordinal
                or identity.get("source_turn_id") != source["turn_id"]
                or execution.get("synthesis_status") != "synthesized"
                or execution.get("turn_audio") != expected_turn_audio
                or persisted.get("labels") != source["labels"]
                or requested.get("required") != source["acoustic"]["required"]
                or requested.get("best_effort") != source["acoustic"].get("best_effort")
                or isinstance(start_sec, bool)
                or not isinstance(start_sec, (int, float))
                or not math.isfinite(start_sec)
                or isinstance(end_sec, bool)
                or not isinstance(end_sec, (int, float))
                or not math.isfinite(end_sec)
                or start_sec < previous_end
                or end_sec <= start_sec
            ):
                return False
            speech_seconds = (
                len(AudioSegment.from_file(out_dir / expected_turn_audio)) / 1000
            )
            if speech_seconds <= 0 or abs(end_sec - start_sec - speech_seconds) > 0.05:
                return False
            previous_end = end_sec

        return all(
            path.is_file() and len(AudioSegment.from_file(path)) > 0
            for path in (out_dir / expected_clean, out_dir / expected_telephone)
        )
    except Exception:  # noqa: BLE001 - corrupt artifacts should trigger rerender
        return False


def _entry_is_resumable(
    previous: dict[str, Any] | None,
    *,
    input_record: InputRecord,
    fingerprint: str,
    output_root: Path,
    sidecar_entry: dict[str, Any],
    engine: str,
    current_backend_identity: dict[str, Any],
    project_root: Path,
) -> bool:
    dialogue_id = input_record.dialogue_id
    candidate = bool(
        previous
        and previous.get("dialogue_id") == dialogue_id
        and previous.get("status") == "success"
        and previous.get("action") in {_RENDERED, _RESUMED}
        and previous.get("render_fingerprint") == fingerprint
        and _resume_artifacts_valid(
            output_root,
            input_record,
            engine=engine,
            current_backend_identity=current_backend_identity,
        )
    )
    if not candidate:
        return False
    roles = sidecar_entry.get("roles")
    if not isinstance(roles, dict):
        return False
    try:
        for logical_role, role in roles.items():
            selected_reference(
                role,
                engine,
                logical_role,
                project_root=project_root,
                verify_file=True,
            )
    except (FinalReferenceError, OSError):
        return False
    return True


def _record_source(record: InputRecord | InputRecordFailure) -> dict[str, Any]:
    source = {
        "container_path": str(record.container_path),
        "format": record.source_format,
        "line_number": record.line_number,
    }
    if isinstance(record, InputRecord):
        source["record_sha256"] = record.record_sha256
    return source


def _batch_status(results: list[dict[str, Any]]) -> str:
    failures = sum(item["status"] == "failed" for item in results)
    if not results or failures == len(results):
        return "failure"
    return "partial_failure" if failures else "success"


async def run_batch(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    config_sha256: str,
    containers: list[Path],
    loaded: list[InputRecord | InputRecordFailure],
    run_started: datetime.datetime,
    batch_started: float,
    logger: logging.Logger,
) -> int:
    """Orchestrate production records without preparing controls before render."""
    engine = config.get("tts", {}).get("engine")
    if engine not in VALID_ENGINES:
        raise RuntimeError(
            "Production batches require tts.engine='higgs' or tts.engine='cosyvoice'"
        )

    records = [item for item in loaded if isinstance(item, InputRecord)]
    output_paths = {
        record.dialogue_id: _safe_dialogue_output_path(args.output, record.dialogue_id)
        for record in records
    }
    policy: ExclusionPolicy | None = (
        load_exclusion_policy(args.exclusion_policy)
        if args.exclusion_policy is not None
        else None
    )
    policy_identity = {
        "schema_version": policy.schema_version if policy else None,
        "policy_sha256": policy.policy_sha256 if policy else None,
    }

    invalid_records: dict[int, Exception] = {}
    valid_records: list[InputRecord] = []
    for record in records:
        try:
            validate_dialogue(record.raw)
        except FinalInputValidationError as error:
            invalid_records[id(record)] = FinalInputValidationError(
                "Production input must use the final upstream dialogue contract; "
                "legacy/v0.2 input is no longer supported by the 5703tts CLI. "
                f"{error}"
            )
        else:
            valid_records.append(record)

    exclusion_by_id = {
        record.dialogue_id: policy.exclusion_for(record.dialogue_id) if policy else None
        for record in valid_records
    }
    eligible = [
        record
        for record in valid_records
        if exclusion_by_id[record.dialogue_id] is None
    ]
    if eligible and args.speaker_sidecar is None:
        raise RuntimeError("Production batches require --speaker-sidecar")
    if args.speaker_sidecar is None:
        sidecar, sidecar_by_id = {}, {}
    else:
        sidecar, sidecar_by_id = _load_sidecar(args.speaker_sidecar)
    missing_sidecar = [
        record.dialogue_id
        for record in eligible
        if record.dialogue_id not in sidecar_by_id
    ]
    if missing_sidecar:
        raise RuntimeError(
            "Final speaker sidecar is missing dialogue_id entries: "
            + ", ".join(sorted(missing_sidecar))
        )

    registry_sha256 = sidecar.get("registry_sha256")
    active_speakers_sha256 = sidecar.get("active_speakers_sha256")
    fingerprints: dict[str, str | None] = {}
    for record in eligible:
        try:
            components = render_fingerprint_components(
                record,
                config=config,
                sidecar_entry=sidecar_by_id[record.dialogue_id],
            )
        except BatchIdentityError:
            fingerprints[record.dialogue_id] = None
        else:
            fingerprints[record.dialogue_id] = render_fingerprint(components)

    manifest_path = args.output / "batch_result.json"
    previous_manifest = (
        _load_previous_batch_result(manifest_path) if args.resume else None
    )
    previous_results = _index_previous_results(previous_manifest)
    batch_backend_identity = backend_identity(config)
    batch_backend_hash = render_fingerprint(batch_backend_identity)

    results: list[dict[str, Any]] = []
    for item in loaded:
        if isinstance(item, InputRecordFailure):
            results.append(
                {
                    "dialogue_id": None,
                    "source": _record_source(item),
                    "action": _INPUT_ERROR,
                    "status": "failed",
                    "render_fingerprint": None,
                    "output_dir": None,
                    "error": {
                        "type": "InputRecordFailure",
                        "message": item.message,
                    },
                    "exclusion": None,
                }
            )
            continue
        if item_id_error := invalid_records.get(id(item)):
            results.append(
                {
                    "dialogue_id": item.dialogue_id,
                    "source": _record_source(item),
                    "action": _INPUT_ERROR,
                    "status": "failed",
                    "render_fingerprint": None,
                    "output_dir": None,
                    "error": {
                        "type": type(item_id_error).__name__,
                        "message": str(item_id_error),
                    },
                    "exclusion": None,
                }
            )
            continue

        exclusion = exclusion_by_id[item.dialogue_id]
        if exclusion is not None:
            results.append(
                {
                    "dialogue_id": item.dialogue_id,
                    "source": _record_source(item),
                    "action": _EXCLUDED,
                    "status": "excluded",
                    "render_fingerprint": None,
                    "output_dir": None,
                    "error": None,
                    "exclusion": {
                        "reason": exclusion.reason,
                        "provenance": exclusion.provenance,
                        "policy_sha256": policy.policy_sha256,
                    },
                }
            )
            continue

        fingerprint = fingerprints[item.dialogue_id]
        expected_out_dir = output_paths[item.dialogue_id]
        previous = previous_results.get(item.dialogue_id)
        if (
            fingerprint is not None
            and args.resume
            and _entry_is_resumable(
                previous,
                input_record=item,
                fingerprint=fingerprint,
                output_root=args.output,
                sidecar_entry=sidecar_by_id[item.dialogue_id],
                engine=engine,
                current_backend_identity=batch_backend_identity,
                project_root=Path.cwd(),
            )
        ):
            action, status, error = _RESUMED, "success", None
        else:
            try:
                _cleanup_managed_dialogue_artifacts(expected_out_dir, item.dialogue_id)
            except (OSError, RuntimeError) as cleanup_error:
                action, status = _RENDER_FAILED, "failed"
                error = {
                    "type": type(cleanup_error).__name__,
                    "message": (
                        f"Failed to clean managed artifacts before rerender: "
                        f"{cleanup_error}"
                    ),
                }
            else:
                pipeline_result = await run_dialogue(
                    item,
                    sidecar,
                    config,
                    args.output,
                    project_root=Path.cwd(),
                )
                if pipeline_result.status == "success":
                    action, status, error = _RENDERED, "success", None
                else:
                    action, status = _RENDER_FAILED, "failed"
                    error = {
                        "type": pipeline_result.error_type or "PipelineFailure",
                        "message": pipeline_result.error or "Unknown dialogue failure",
                    }
        results.append(
            {
                "dialogue_id": item.dialogue_id,
                "source": _record_source(item),
                "action": action,
                "status": status,
                "render_fingerprint": fingerprint,
                "output_dir": str(expected_out_dir),
                "error": error,
                "exclusion": None,
            }
        )

    finished_at = datetime.datetime.now().astimezone()
    status = _batch_status(results)
    rendered = sum(item["action"] == _RENDERED for item in results)
    resumed = sum(item["action"] == _RESUMED for item in results)
    excluded = sum(item["action"] == _EXCLUDED for item in results)
    failed = sum(item["status"] == "failed" for item in results)
    input_failures = sum(item["action"] == _INPUT_ERROR for item in results)
    mapping = batch_backend_identity["control_mapping"]
    batch_result = {
        "manifest_version": BATCH_MANIFEST_VERSION,
        "status": status,
        "started_at": run_started.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_seconds": round((finished_at - run_started).total_seconds(), 3),
        "backend": engine,
        "backend_identity": batch_backend_identity,
        "backend_identity_sha256": batch_backend_hash,
        "config_path": str(args.config),
        "config_sha256": config_sha256,
        "input_root": str(args.input),
        "output_root": str(args.output),
        "resume_requested": args.resume,
        "batch_provenance": {
            "controlled_tts": {
                "mapping_name": mapping.get("name", mapping.get("mapping_version")),
                "mapping_version": mapping.get(
                    "version", mapping.get("mapping_version")
                ),
                "release_status": mapping.get("status", mapping.get("release_status")),
                **(
                    {"contract_sha256": mapping["provenance"]["contract_sha256"]}
                    if "provenance" in mapping
                    else {}
                ),
                "implementation_id": mapping["implementation_id"],
            },
            "exclusion_policy": policy_identity,
            "speaker_sidecar": {
                "path": str(args.speaker_sidecar),
                "assignment_sha256": sidecar.get("assignment_sha256"),
                "registry_sha256": registry_sha256,
                "active_speakers_sha256": active_speakers_sha256,
            },
            "input_containers": [str(path) for path in containers],
        },
        "records_total": len(results),
        "dialogues_total": len(records),
        "dialogues_succeeded": rendered + resumed,
        "dialogues_failed": failed,
        "dialogues_rendered": rendered,
        "dialogues_resumed": resumed,
        "dialogues_excluded": excluded,
        "input_failures": input_failures,
        "dialogues_skipped": resumed,
        "summary": {
            "total_valid_records": len(records),
            "rendered": rendered,
            "resumed": resumed,
            "excluded": excluded,
            "failed": failed,
            "input_failures": input_failures,
        },
        "results": results,
    }
    written = _write_batch_result(batch_result, args.output)
    logger.info(
        "event=final_batch_complete status=%s rendered=%d resumed=%d excluded=%d "
        "failed=%d elapsed_sec=%.2f manifest=%s",
        status,
        rendered,
        resumed,
        excluded,
        failed,
        time.perf_counter() - batch_started,
        written,
    )
    return 0 if status == "success" else 1
