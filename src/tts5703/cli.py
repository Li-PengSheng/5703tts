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

from .config import load_config
from .cosyvoice_controls import (
    COSYVOICE_CONTROL_MAPPING_NAME,
    COSYVOICE_CONTROL_MAPPING_STATUS,
    COSYVOICE_CONTROL_MAPPING_VERSION,
)
from .pipeline import PipelineResult, run_dialogue
from .qc import run_qc
from .tts_engine import describe_engine
from .validate import ValidationError, load_and_validate

_RENDERED = "rendered"
_RETRIED = "retried"
_SKIPPED_COMPLETED = "skipped_completed"
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
    engine = describe_engine(config)
    identity = {"backend": engine["engine"]}
    if model := engine.get("model"):
        identity["model"] = model
    if engine["engine"] == "cosyvoice":
        identity["control_mapping"] = {
            "name": COSYVOICE_CONTROL_MAPPING_NAME,
            "version": COSYVOICE_CONTROL_MAPPING_VERSION,
            "status": COSYVOICE_CONTROL_MAPPING_STATUS,
        }
    return identity


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
        out_dir = output_root / dialogue.dialogue_id
        if (
            previous.get("dialogue_id") != dialogue.dialogue_id
            or not isinstance(previous.get("output_dir"), str)
            or Path(previous["output_dir"]).resolve() != out_dir.resolve()
        ):
            return None
        metadata_path = out_dir / f"{dialogue.dialogue_id}_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            not isinstance(metadata, dict)
            or metadata.get("dialogue_id") != dialogue.dialogue_id
            or metadata.get("clean_audio") != f"{dialogue.dialogue_id}_clean.wav"
            or metadata.get("telephone_audio")
            != f"{dialogue.dialogue_id}_telephone.wav"
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


async def main() -> int:
    """Discover input JSON files and render them sequentially as one batch.

    ``run_dialogue`` converts per-dialogue failures into ``PipelineResult`` values,
    which lets this loop continue with later files. Configuration failures, malformed
    resume manifests, and duplicate current ``dialogue_id`` values abort before any
    dialogue is rendered and do not write ``batch_result.json``.
    """
    parser = argparse.ArgumentParser(
        description="Render crisis dialogue JSON files with a configured TTS engine."
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
        config.get("tts", {}).get("engine", "edge_tts"),
    )
    config_sha256 = _sha256(args.config)
    backend_identity = _backend_identity(config)
    manifest_path = args.output / "batch_result.json"
    try:
        previous_manifest = (
            _load_previous_batch_result(manifest_path) if args.resume else None
        )
        previous_results = _index_previous_results(previous_manifest)
    except RuntimeError:
        logger.exception(
            "event=batch_failed stage=resume_manifest path=%s", manifest_path
        )
        raise
    if args.resume and previous_manifest is None:
        logger.info("event=resume_no_previous_manifest path=%s", manifest_path)
    previous_batch_compatible = bool(
        previous_manifest
        and previous_manifest.get("config_sha256") == config_sha256
        and previous_manifest.get("backend_identity") == backend_identity
    )
    json_files = sorted(args.input.glob("*.json"))
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
