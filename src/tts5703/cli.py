"""Command-line entry point for the production rendering pipeline."""

import argparse
import asyncio
import datetime
import hashlib
import logging
import time
from pathlib import Path

from .batch import run_batch
from .config import VALID_ENGINES, load_config
from .input.records import InputRecord, InputRecordFailure, read_input_records


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


def _sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


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
    return await run_batch(
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
