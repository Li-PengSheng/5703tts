"""Batch command-line entry point (stage 7 logging)."""

import argparse
import asyncio
import datetime
import logging
import time
from pathlib import Path

from .config import load_config
from .pipeline import run_dialogue


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


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render crisis dialogue JSON files with a configured TTS engine."
    )
    parser.add_argument("--input", type=Path, default=Path("data/input"))
    parser.add_argument("--output", type=Path, default=Path("data/output"))
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument(
        "--log-dir", type=Path, default=Path("logs"), help="Log directory (default: logs)"
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
        return
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
        results.append(result)
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
    success_count = sum(result.status == "success" for result in results)
    logger.info(
        "event=batch_complete success=%d failed=%d total=%d elapsed_sec=%.2f log=%s",
        success_count,
        len(results) - success_count,
        len(results),
        time.perf_counter() - batch_started,
        log_path,
    )


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
