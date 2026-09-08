"""Batch command-line entry point and authoritative result manifest."""

import argparse
import asyncio
import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any

from .config import load_config
from .pipeline import PipelineResult, run_dialogue


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
    backend: str,
    results: list[tuple[Path, PipelineResult]],
) -> dict[str, Any]:
    """Build the one result object used for JSON output and process status."""
    succeeded = sum(result.status == "success" for _, result in results)
    failed = len(results) - succeeded
    if succeeded == len(results) and results:
        status = "success"
    elif succeeded:
        status = "partial_failure"
    else:
        status = "failure"

    dialogue_results = []
    for input_path, result in results:
        item: dict[str, Any] = {
            "dialogue_id": result.dialogue_id,
            "input_path": str(input_path),
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
        "backend": backend,
        "config_path": str(config_path),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "dialogues_total": len(results),
        "dialogues_succeeded": succeeded,
        "dialogues_failed": failed,
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


async def main() -> int:
    """Discover input JSON files and render them sequentially as one batch.

    ``run_dialogue`` converts per-dialogue failures into ``PipelineResult`` values,
    which lets this loop continue with later files. Configuration failures happen
    outside that boundary and intentionally abort the whole invocation.
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
    json_files = sorted(args.input.glob("*.json"))
    if not json_files:
        logger.warning("event=batch_no_input input=%s", args.input)
    else:
        logger.info("event=input_discovered dialogue_count=%d", len(json_files))

    results = []
    for index, path in enumerate(json_files, start=1):
        logger.info(
            "event=dialogue_queued index=%d total=%d source=%s",
            index,
            len(json_files),
            path.name,
        )
        started = time.perf_counter()
        result = await run_dialogue(path, config, args.output)
        elapsed = time.perf_counter() - started
        results.append((path, result))
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
        backend=config.get("tts", {}).get("engine", "edge_tts"),
        results=results,
    )
    manifest_path = _write_batch_result(batch_result, args.output)
    logger.info(
        "event=batch_complete status=%s success=%d failed=%d total=%d "
        "elapsed_sec=%.2f manifest=%s log=%s",
        batch_result["status"],
        batch_result["dialogues_succeeded"],
        batch_result["dialogues_failed"],
        batch_result["dialogues_total"],
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
