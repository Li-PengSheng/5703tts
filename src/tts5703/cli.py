"""Batch command-line entry point and authoritative result manifest."""

import argparse
import asyncio
import datetime
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from .batch_identity import (
    FINAL_BATCH_MANIFEST_VERSION,
    BatchIdentityError,
    final_backend_identity,
    render_fingerprint,
    render_fingerprint_components,
)
from .config import VALID_ENGINES, load_config
from .exclusion_policy import ExclusionPolicy, load_exclusion_policy
from .final_input import FinalInputValidationError, validate_final_dialogue
from .final_references import FinalReferenceError, selected_reference
from .input_records import InputRecord, InputRecordFailure, read_input_records
from .pipeline import run_dialogue

_RENDERED = "rendered"
_FINAL_RESUMED = "resumed"
_FINAL_EXCLUDED = "excluded_known_issue"
_FINAL_INPUT_ERROR = "input_error"
_FINAL_RENDER_FAILED = "render_failed"
_RESUME_MANIFEST_HINT = "Repair or remove this file, or run without --resume to ignore previous batch state."


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
        or manifest.get("manifest_version") != FINAL_BATCH_MANIFEST_VERSION
        or manifest.get("status") not in {"success", "partial_failure", "failure"}
        or not isinstance(manifest.get("results"), list)
    ):
        raise RuntimeError(
            f"Cannot resume from malformed batch result: {path}. {_RESUME_MANIFEST_HINT}"
        )
    return manifest


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


async def _run_production_batch(
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
    final_records: list[InputRecord] = []
    for record in records:
        try:
            validate_final_dialogue(record.raw)
        except FinalInputValidationError as error:
            invalid_records[id(record)] = FinalInputValidationError(
                "Production input must use the final upstream dialogue contract; "
                "legacy/v0.2 input is no longer supported by the 5703tts CLI. "
                f"{error}"
            )
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
    if eligible and args.speaker_sidecar is None:
        raise RuntimeError("Production batches require --speaker-sidecar")
    if args.speaker_sidecar is None:
        sidecar, sidecar_by_id = {}, {}
    else:
        sidecar, sidecar_by_id = _load_final_sidecar(args.speaker_sidecar)
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
                action, status = _FINAL_RENDER_FAILED, "failed"
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
    status = _final_status(results)
    rendered = sum(item["action"] == _RENDERED for item in results)
    resumed = sum(item["action"] == _FINAL_RESUMED for item in results)
    excluded = sum(item["action"] == _FINAL_EXCLUDED for item in results)
    failed = sum(item["status"] == "failed" for item in results)
    input_failures = sum(item["action"] == _FINAL_INPUT_ERROR for item in results)
    mapping = batch_backend_identity["control_mapping"]
    batch_result = {
        "manifest_version": FINAL_BATCH_MANIFEST_VERSION,
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
    """Run the one production JSON/JSONL pipeline."""
    parser = argparse.ArgumentParser(
        description="Render final upstream dialogue JSON/JSONL."
    )
    parser.add_argument("--input", type=Path, default=Path("data/input"))
    parser.add_argument("--output", type=Path, default=Path("data/output"))
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only dialogues with matching semantic fingerprints",
    )
    parser.add_argument(
        "--speaker-sidecar",
        type=Path,
        help="Production speaker-assignment sidecar",
    )
    parser.add_argument(
        "--exclusion-policy",
        type=Path,
        help="Optional production dialogue exclusion policy",
    )
    args = parser.parse_args()
    run_started = datetime.datetime.now().astimezone()
    log_path = args.log_dir / f"run_{run_started:%Y-%m-%d}.log"
    configure_logging(log_path, args.verbose)
    logger = logging.getLogger(__name__)
    batch_started = time.perf_counter()

    try:
        config = load_config(args.config)
    except Exception:
        logger.exception("event=batch_failed stage=config_load")
        raise
    engine = config.get("tts", {}).get("engine")
    if engine not in VALID_ENGINES:
        raise RuntimeError(
            "Production 5703tts supports only tts.engine='higgs' or "
            "tts.engine='cosyvoice'"
        )

    config_sha256 = _sha256(args.config)
    containers = _discover_cli_containers(args.input)
    loaded = _read_cli_records(containers)
    records = [item for item in loaded if isinstance(item, InputRecord)]
    _require_unique_record_ids(records)
    return await _run_production_batch(
        args=args,
        config=config,
        config_sha256=config_sha256,
        containers=containers,
        loaded=loaded,
        run_started=run_started,
        batch_started=batch_started,
        logger=logger,
    )


def run() -> None:
    """Synchronous console-script wrapper for the asynchronous batch entry point."""
    raise SystemExit(asyncio.run(main()))


if __name__ == "__main__":
    run()
