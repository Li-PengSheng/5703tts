"""Batch command-line entry point and authoritative result manifest."""

import argparse
import asyncio
import datetime
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backend_info import backend_identity
from .batch_identity import (
    FINAL_BATCH_MANIFEST_VERSION,
    BatchIdentityError,
    final_backend_identity,
    render_fingerprint,
    render_fingerprint_components,
)
from .config import load_config
from .exclusion_policy import ExclusionPolicy, load_exclusion_policy
from .final_input import (
    FinalInputValidationError,
    SchemaFamily,
    SchemaFamilyError,
    detect_schema_family,
    validate_final_dialogue,
)
from .final_references import FinalReferenceError, selected_reference
from .input_records import InputRecord, InputRecordFailure, read_input_records
from .pipeline import PipelineResult, run_dialogue, run_final_dialogue
from .qc import run_qc
from .validate import ValidationError, load_and_validate

_RENDERED = "rendered"
_RETRIED = "retried"
_SKIPPED_COMPLETED = "skipped_completed"
_FINAL_RESUMED = "resumed"
_FINAL_EXCLUDED = "excluded_known_issue"
_FINAL_INPUT_ERROR = "input_error"
_FINAL_RENDER_FAILED = "render_failed"
_RESUME_MANIFEST_HINT = "Repair or remove this file, or run without --resume to ignore previous batch state."


@dataclass(frozen=True)
class DialogueBatchResult:
    input_path: Path
    input_sha256: str | None
    action: str
    result: PipelineResult


def configure_logging(log_path: Path, verbose: bool) -> None:
    """Write detailed logs to file and concise progress to the console."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.DEBUG, handlers=[file_handler, console_handler], force=True
    )


def _build_batch_result(
    *,
    started_at: datetime.datetime,
    finished_at: datetime.datetime,
    input_root: Path,
    output_root: Path,
    config_path: Path,
    backend_identity: dict[str, Any],
    config_sha256: str,
    resume_requested: bool,
    results: list[DialogueBatchResult],
) -> dict[str, Any]:
    """Build the one result object used for JSON output and process status."""
    succeeded = sum(item.result.status == "success" for item in results)
    failed = len(results) - succeeded
    if succeeded == len(results) and results:
        status = "success"
    elif succeeded:
        status = "partial_failure"
    else:
        status = "failure"

    dialogue_results = []
    for batch_item in results:
        result = batch_item.result
        item: dict[str, Any] = {
            "dialogue_id": result.dialogue_id,
            "input_path": str(batch_item.input_path),
            "input_sha256": batch_item.input_sha256,
            "action": batch_item.action,
            "status": result.status,
            "output_dir": str(result.out_dir) if result.out_dir else None,
        }
        if result.status != "success":
            item["error"] = {
                "type": result.error_type or "PipelineFailure",
                "message": result.error or "Unknown dialogue failure",
            }
        dialogue_results.append(item)

    return {
        "status": status,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "backend": backend_identity["backend"],
        "backend_identity": backend_identity,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "resume_requested": resume_requested,
        "dialogues_total": len(results),
        "dialogues_succeeded": succeeded,
        "dialogues_failed": failed,
        "dialogues_rendered": sum(
            item.action != _SKIPPED_COMPLETED for item in results
        ),
        "dialogues_retried": sum(item.action == _RETRIED for item in results),
        "dialogues_skipped": sum(item.action == _SKIPPED_COMPLETED for item in results),
        "results": dialogue_results,
    }


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


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _backend_identity(config: dict[str, Any]) -> dict[str, Any]:
    return backend_identity(config)


def _identity_allows_resume(identity: Any) -> bool:
    return isinstance(identity, dict) and (
        identity.get("backend") != "higgs" or identity.get("identity_complete") is True
    )


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
        isinstance(manifest, dict)
        and manifest.get("manifest_version") == FINAL_BATCH_MANIFEST_VERSION
    ):
        if manifest.get("status") not in {
            "success",
            "partial_failure",
            "failure",
        } or not isinstance(manifest.get("results"), list):
            raise RuntimeError(
                f"Cannot resume from malformed batch result: {path}. "
                f"{_RESUME_MANIFEST_HINT}"
            )
        return manifest
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") not in {"success", "partial_failure", "failure"}
        or not isinstance(manifest.get("results"), list)
    ):
        raise RuntimeError(
            f"Cannot resume from malformed batch result: {path}. {_RESUME_MANIFEST_HINT}"
        )
    for result in manifest["results"]:
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("input_path"), str)
            or result.get("status") not in {"success", "failed"}
        ):
            raise RuntimeError(
                f"Cannot resume from malformed batch result: {path}. "
                f"{_RESUME_MANIFEST_HINT}"
            )
    return manifest


def _index_previous_results(manifest: dict[str, Any] | None) -> dict[Path, dict]:
    indexed = {}
    for result in manifest["results"] if manifest else []:
        input_path = Path(result["input_path"]).resolve()
        if input_path in indexed:
            raise RuntimeError(
                "Cannot resume: duplicate input path in previous batch result: "
                f"{input_path}. {_RESUME_MANIFEST_HINT}"
            )
        indexed[input_path] = result
    return indexed


def _duplicate_dialogue_id_error(duplicates: dict[str, list[Path]]) -> RuntimeError:
    lines = [
        (
            "Duplicate dialogue_id values were found in this batch. "
            "Each dialogue_id must be unique because outputs are written to "
            "<output>/<dialogue_id>/."
        )
    ]
    for dialogue_id in sorted(duplicates):
        lines.append(f"Duplicate dialogue_id {dialogue_id!r} found in:")
        lines.extend(f"- {path}" for path in duplicates[dialogue_id])
    return RuntimeError("\n".join(lines))


def _require_unique_dialogue_ids(
    json_files: list[Path], config: dict[str, Any]
) -> None:
    """Reject the batch if two current inputs share a canonical dialogue_id."""
    by_id: dict[str, list[Path]] = {}
    for path in json_files:
        try:
            dialogue = load_and_validate(path, config)
        except ValidationError:
            # Per-dialogue validation failures stay in the render loop. A file
            # that cannot yield a canonical ID cannot occupy an output directory.
            continue
        by_id.setdefault(dialogue.dialogue_id, []).append(path)
    duplicates = {
        dialogue_id: paths for dialogue_id, paths in by_id.items() if len(paths) > 1
    }
    if duplicates:
        raise _duplicate_dialogue_id_error(duplicates)


def _verified_completed_result(
    input_path: Path,
    previous: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
) -> PipelineResult | None:
    """Return a successful result only when existing production QC still passes."""
    try:
        dialogue = load_and_validate(input_path, config)
        current_backend_identity = backend_identity(config)
        out_dir = output_root / dialogue.dialogue_id
        if (
            previous.get("dialogue_id") != dialogue.dialogue_id
            or not isinstance(previous.get("output_dir"), str)
            or Path(previous["output_dir"]).resolve() != out_dir.resolve()
        ):
            return None
        metadata_path = out_dir / f"{dialogue.dialogue_id}_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_tts = metadata.get("tts") if isinstance(metadata, dict) else None
        metadata_backend_identity = (
            metadata_tts.get("backend_identity")
            if isinstance(metadata_tts, dict)
            else None
        )
        if (
            not isinstance(metadata, dict)
            or metadata.get("dialogue_id") != dialogue.dialogue_id
            or metadata.get("clean_audio") != f"{dialogue.dialogue_id}_clean.wav"
            or metadata.get("telephone_audio")
            != f"{dialogue.dialogue_id}_telephone.wav"
            or not isinstance(metadata_tts, dict)
            or metadata_tts.get("engine") != current_backend_identity["backend"]
            or metadata_backend_identity != current_backend_identity
            or not _identity_allows_resume(current_backend_identity)
            or not _identity_allows_resume(metadata_backend_identity)
        ):
            return None
        qc = run_qc(
            dialogue,
            out_dir,
            out_dir / metadata["clean_audio"],
            out_dir / metadata["telephone_audio"],
            metadata,
        )
    except Exception:  # noqa: BLE001 - any unreadable prior artifact means rerender
        return None
    if not qc.passed:
        return None
    return PipelineResult(dialogue.dialogue_id, "success", qc=qc, out_dir=out_dir)


def _discover_cli_containers(input_path: Path) -> list[Path]:
    """Discover the final-capable top-level JSON/JSONL candidate set."""
    if input_path.is_file():
        if input_path.suffix.lower() not in {".json", ".jsonl"}:
            raise RuntimeError(f"Unsupported input file type: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        return []
    return sorted(
        (
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
        ),
        key=lambda path: path.as_posix(),
    )


def _discover_legacy_json_files(input_path: Path) -> list[Path]:
    """Preserve the legacy renderer's explicit/top-level JSON-only input set."""
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".json" else []
    if not input_path.is_dir():
        return []
    return sorted(input_path.glob("*.json"))


def _read_cli_records(
    containers: list[Path],
) -> list[InputRecord | InputRecordFailure]:
    records: list[InputRecord | InputRecordFailure] = []
    for path in containers:
        records.extend(read_input_records(path))
    return records


def _require_unique_record_ids(records: list[InputRecord]) -> None:
    locations: dict[str, list[str]] = {}
    for record in records:
        location = str(record.container_path)
        if record.line_number is not None:
            location += f":{record.line_number}"
        locations.setdefault(record.dialogue_id, []).append(location)
    duplicates = {
        dialogue_id: values
        for dialogue_id, values in locations.items()
        if len(values) > 1
    }
    if duplicates:
        lines = ["Duplicate dialogue_id values were found in this final batch."]
        for dialogue_id in sorted(duplicates):
            lines.append(f"Duplicate dialogue_id {dialogue_id!r} found in:")
            lines.extend(f"- {location}" for location in duplicates[dialogue_id])
        raise RuntimeError("\n".join(lines))


def _is_final_batch(input_path: Path, records: list[InputRecord]) -> bool:
    families: set[SchemaFamily] = set()
    for record in records:
        try:
            families.add(detect_schema_family(record.raw))
        except SchemaFamilyError:
            continue
    if SchemaFamily.FINAL_NESTED in families and len(families) > 1:
        raise RuntimeError(
            "Cannot mix final nested and legacy/v0.2 records in one batch"
        )
    if SchemaFamily.FINAL_NESTED in families:
        return True
    return input_path.is_file() and input_path.suffix.lower() == ".jsonl"


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
    root = output_root.resolve()
    candidate = (root / dialogue_id).resolve()
    if candidate.parent != root or not candidate.is_relative_to(root):
        raise RuntimeError(f"Final dialogue_id escapes output root: {dialogue_id!r}")
    return candidate


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


def _load_final_sidecar(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
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


def _index_previous_final_results(
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


def _final_artifacts_exist(output_root: Path, dialogue_id: str) -> bool:
    out_dir = _safe_dialogue_output_path(output_root, dialogue_id)
    return all(
        (out_dir / filename).is_file()
        for filename in (
            f"{dialogue_id}_clean.wav",
            f"{dialogue_id}_telephone.wav",
            f"{dialogue_id}_metadata.json",
        )
    )


def _final_entry_is_resumable(
    previous: dict[str, Any] | None,
    *,
    dialogue_id: str,
    fingerprint: str,
    output_root: Path,
    sidecar_entry: dict[str, Any],
    engine: str,
    project_root: Path,
) -> bool:
    candidate = bool(
        previous
        and previous.get("dialogue_id") == dialogue_id
        and previous.get("status") == "success"
        and previous.get("action") in {_RENDERED, _FINAL_RESUMED}
        and previous.get("render_fingerprint") == fingerprint
        and _final_artifacts_exist(output_root, dialogue_id)
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


def _final_status(results: list[dict[str, Any]]) -> str:
    failures = sum(item["status"] == "failed" for item in results)
    if not results or failures == len(results):
        return "failure"
    return "partial_failure" if failures else "success"


async def _run_final_batch(
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
    """Orchestrate final records without preparing controls before rendering."""
    engine = config.get("tts", {}).get("engine")
    if engine not in {"higgs", "cosyvoice"}:
        raise RuntimeError(
            "Final batches require tts.engine='higgs' or tts.engine='cosyvoice'"
        )
    if args.speaker_sidecar is None:
        raise RuntimeError("Final batches require --speaker-sidecar")

    records = [item for item in loaded if isinstance(item, InputRecord)]
    output_paths = {
        record.dialogue_id: _safe_dialogue_output_path(args.output, record.dialogue_id)
        for record in records
    }
    sidecar, sidecar_by_id = _load_final_sidecar(args.speaker_sidecar)
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
    final_records: list[InputRecord] = []
    for record in records:
        try:
            if detect_schema_family(record.raw) is not SchemaFamily.FINAL_NESTED:
                raise SchemaFamilyError("record is not final nested")
            validate_final_dialogue(record.raw)
        except (SchemaFamilyError, FinalInputValidationError) as error:
            invalid_records[id(record)] = error
        else:
            final_records.append(record)

    exclusion_by_id = {
        record.dialogue_id: policy.exclusion_for(record.dialogue_id) if policy else None
        for record in final_records
    }
    eligible = [
        record
        for record in final_records
        if exclusion_by_id[record.dialogue_id] is None
    ]
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
    previous_results = _index_previous_final_results(previous_manifest)
    batch_backend_identity = final_backend_identity(config)
    batch_backend_hash = render_fingerprint(batch_backend_identity)

    results: list[dict[str, Any]] = []
    for item in loaded:
        if isinstance(item, InputRecordFailure):
            results.append(
                {
                    "dialogue_id": None,
                    "schema_family": None,
                    "source": _record_source(item),
                    "action": _FINAL_INPUT_ERROR,
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
                    "schema_family": None,
                    "source": _record_source(item),
                    "action": _FINAL_INPUT_ERROR,
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
                    "schema_family": SchemaFamily.FINAL_NESTED.value,
                    "source": _record_source(item),
                    "action": _FINAL_EXCLUDED,
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
            and _final_entry_is_resumable(
                previous,
                dialogue_id=item.dialogue_id,
                fingerprint=fingerprint,
                output_root=args.output,
                sidecar_entry=sidecar_by_id[item.dialogue_id],
                engine=engine,
                project_root=Path.cwd(),
            )
        ):
            action, status, error = _FINAL_RESUMED, "success", None
        else:
            pipeline_result = await run_final_dialogue(
                item,
                sidecar,
                config,
                args.output,
                project_root=Path.cwd(),
            )
            if pipeline_result.status == "success":
                action, status, error = _RENDERED, "success", None
            else:
                action, status = _FINAL_RENDER_FAILED, "failed"
                error = {
                    "type": pipeline_result.error_type or "PipelineFailure",
                    "message": pipeline_result.error or "Unknown dialogue failure",
                }
        results.append(
            {
                "dialogue_id": item.dialogue_id,
                "schema_family": SchemaFamily.FINAL_NESTED.value,
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
    status = _final_status(results)
    rendered = sum(item["action"] == _RENDERED for item in results)
    resumed = sum(item["action"] == _FINAL_RESUMED for item in results)
    excluded = sum(item["action"] == _FINAL_EXCLUDED for item in results)
    failed = sum(item["status"] == "failed" for item in results)
    input_failures = sum(item["action"] == _FINAL_INPUT_ERROR for item in results)
    mapping = batch_backend_identity["control_mapping"]
    batch_result = {
        "manifest_version": FINAL_BATCH_MANIFEST_VERSION,
        "batch_schema_family": SchemaFamily.FINAL_NESTED.value,
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


async def main() -> int:
    """Discover supported input records and render them sequentially as one batch.

    ``run_dialogue`` converts per-dialogue failures into ``PipelineResult`` values,
    which lets this loop continue with later files. Configuration failures, malformed
    resume manifests, and duplicate current ``dialogue_id`` values abort before any
    dialogue is rendered and do not write ``batch_result.json``.
    """
    parser = argparse.ArgumentParser(
        description="Render crisis dialogue JSON/JSONL with a configured TTS engine."
    )
    parser.add_argument("--input", type=Path, default=Path("data/input"))
    parser.add_argument("--output", type=Path, default=Path("data/output"))
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Log directory (default: logs)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show per-turn debug logs in the console"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip compatible completed dialogues and retry incomplete ones",
    )
    parser.add_argument(
        "--speaker-sidecar",
        type=Path,
        help="Final-schema speaker materialization manifest",
    )
    parser.add_argument(
        "--exclusion-policy",
        type=Path,
        help="Optional final-schema dialogue exclusion policy",
    )
    args = parser.parse_args()
    run_started = datetime.datetime.now().astimezone()
    log_path = args.log_dir / f"run_{run_started:%Y-%m-%d}.log"
    configure_logging(log_path, args.verbose)
    logger = logging.getLogger(__name__)
    batch_started = time.perf_counter()

    logger.info("=" * 72)
    logger.info(
        "event=batch_start started_at=%s", run_started.isoformat(timespec="seconds")
    )
    logger.info(
        "event=batch_config input=%s output=%s config=%s log=%s",
        args.input,
        args.output,
        args.config,
        log_path,
    )
    try:
        config = load_config(args.config)
    except Exception:
        logger.exception("event=batch_failed stage=config_load")
        raise
    logger.info(
        "event=engine_selected engine=%s",
        config["tts"]["engine"],
    )
    config_sha256 = _sha256(args.config)
    containers = _discover_cli_containers(args.input)
    loaded_records = _read_cli_records(containers)
    parsed_records = [item for item in loaded_records if isinstance(item, InputRecord)]
    try:
        final_batch = _is_final_batch(args.input, parsed_records)
        if final_batch:
            _require_unique_record_ids(parsed_records)
    except RuntimeError:
        logger.exception("event=batch_failed stage=final_batch_validation")
        raise
    if final_batch:
        return await _run_final_batch(
            args=args,
            config=config,
            config_sha256=config_sha256,
            containers=containers,
            loaded=loaded_records,
            run_started=run_started,
            batch_started=batch_started,
            logger=logger,
        )

    backend_identity = _backend_identity(config)
    manifest_path = args.output / "batch_result.json"
    try:
        previous_manifest = (
            _load_previous_batch_result(manifest_path) if args.resume else None
        )
        legacy_previous_manifest = (
            None
            if previous_manifest
            and previous_manifest.get("manifest_version")
            == FINAL_BATCH_MANIFEST_VERSION
            else previous_manifest
        )
        previous_results = _index_previous_results(legacy_previous_manifest)
    except RuntimeError:
        logger.exception(
            "event=batch_failed stage=resume_manifest path=%s", manifest_path
        )
        raise
    if args.resume and previous_manifest is None:
        logger.info("event=resume_no_previous_manifest path=%s", manifest_path)
    previous_batch_compatible = bool(
        legacy_previous_manifest
        and legacy_previous_manifest.get("config_sha256") == config_sha256
        and legacy_previous_manifest.get("backend_identity") == backend_identity
        and _identity_allows_resume(backend_identity)
        and _identity_allows_resume(legacy_previous_manifest.get("backend_identity"))
    )
    json_files = _discover_legacy_json_files(args.input)
    if not json_files:
        logger.warning("event=batch_no_input input=%s", args.input)
    else:
        logger.info("event=input_discovered dialogue_count=%d", len(json_files))
    try:
        _require_unique_dialogue_ids(json_files, config)
    except RuntimeError:
        logger.exception("event=batch_failed stage=duplicate_dialogue_id")
        raise

    results: list[DialogueBatchResult] = []
    for index, path in enumerate(json_files, start=1):
        try:
            input_sha256 = _sha256(path)
        except OSError:
            input_sha256 = None
        previous = previous_results.get(path.resolve())
        same_previous_input = bool(
            previous and input_sha256 and previous.get("input_sha256") == input_sha256
        )
        action = _RENDERED
        result = None
        if args.resume and previous_batch_compatible and same_previous_input:
            if previous["status"] == "success":
                result = _verified_completed_result(path, previous, config, args.output)
                action = _SKIPPED_COMPLETED if result else _RETRIED
            else:
                action = _RETRIED
        logger.info(
            "event=dialogue_queued index=%d total=%d source=%s action=%s",
            index,
            len(json_files),
            path.name,
            action,
        )
        started = time.perf_counter()
        if result is None:
            result = await run_dialogue(path, config, args.output)
        elapsed = time.perf_counter() - started
        results.append(DialogueBatchResult(path, input_sha256, action, result))
        if result.status == "success":
            logger.info(
                "event=dialogue_complete id=%s status=success elapsed_sec=%.2f output=%s qc=%s",
                result.dialogue_id,
                elapsed,
                result.out_dir,
                result.qc.checks if result.qc else None,
            )
        else:
            logger.error(
                "event=dialogue_complete id=%s status=failed elapsed_sec=%.2f error=%s qc_issues=%s",
                result.dialogue_id,
                elapsed,
                result.error,
                result.qc.issues if result.qc else None,
            )
    finished_at = datetime.datetime.now().astimezone()
    batch_result = _build_batch_result(
        started_at=run_started,
        finished_at=finished_at,
        input_root=args.input,
        output_root=args.output,
        config_path=args.config,
        backend_identity=backend_identity,
        config_sha256=config_sha256,
        resume_requested=args.resume,
        results=results,
    )
    manifest_path = _write_batch_result(batch_result, args.output)
    logger.info(
        "event=batch_complete status=%s success=%d failed=%d total=%d "
        "rendered=%d retried=%d skipped=%d elapsed_sec=%.2f manifest=%s log=%s",
        batch_result["status"],
        batch_result["dialogues_succeeded"],
        batch_result["dialogues_failed"],
        batch_result["dialogues_total"],
        batch_result["dialogues_rendered"],
        batch_result["dialogues_retried"],
        batch_result["dialogues_skipped"],
        time.perf_counter() - batch_started,
        manifest_path,
        log_path,
    )
    return 0 if batch_result["status"] == "success" else 1


def run() -> None:
    """Synchronous console-script wrapper for the asynchronous batch entry point."""
    raise SystemExit(asyncio.run(main()))


if __name__ == "__main__":
    run()
