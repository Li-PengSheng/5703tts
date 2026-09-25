"""Build a retry input pack from quality rejects and frozen source artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from tts5703.input.records import (
    InputRecordFailure,
    canonical_json_sha256,
    read_input_records,
)

_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_OUTPUT_NAMES = ("input.jsonl", "assignments.jsonl", "retry_manifest.json")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _jsonl_rows(path: Path, data: bytes) -> list[tuple[dict[str, Any], bytes]]:
    if path.suffix.lower() != ".jsonl":
        raise ValueError(f"Expected a JSONL file: {path}")
    rows = []
    for line_number, line in enumerate(data.splitlines(keepends=True), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError(
                f"Invalid JSONL row at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise TypeError(f"Expected JSON object at {path}:{line_number}")
        rows.append((row, line))
    return rows


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _line_bytes(line: bytes) -> bytes:
    return line if line.endswith(b"\n") else line + b"\n"


def _stage(output: Path, data: bytes) -> Path:
    path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output, prefix=".retry_", delete=False
        ) as file:
            path = Path(file.name)
            file.write(data)
    except BaseException:
        if path is not None:
            path.unlink(missing_ok=True)
        raise
    return path


def build_retry_batch(
    rejects: Path, corpus: Path, assignments: Path, output: Path, attempt: int
) -> dict[str, Any]:
    if attempt < 1:
        raise ValueError("--attempt must be a positive integer")
    paths = [output / name for name in _OUTPUT_NAMES]
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite retry artifact: {path}")

    rejects_data = rejects.read_bytes()
    corpus_data = corpus.read_bytes()
    assignments_data = assignments.read_bytes()
    if corpus.suffix.lower() != ".jsonl":
        raise ValueError(f"Expected a JSONL corpus: {corpus}")
    queue = []
    seen = set()
    for row, _ in _jsonl_rows(rejects, rejects_data):
        dialogue_id = row.get("dialogue_id")
        source = row.get("source")
        reason_codes = row.get("reason_codes")
        if not isinstance(dialogue_id, str) or not dialogue_id.strip():
            raise ValueError("Retry queue has a blank dialogue_id")
        if dialogue_id in seen:
            raise ValueError(f"Duplicate retry dialogue_id: {dialogue_id}")
        if not isinstance(row.get("render_fingerprint"), str) or not _SHA256.fullmatch(
            row["render_fingerprint"]
        ):
            raise ValueError(f"Invalid render_fingerprint for {dialogue_id}")
        if (
            not isinstance(reason_codes, list)
            or not reason_codes
            or any(
                not isinstance(code, str) or not code.strip() for code in reason_codes
            )
        ):
            raise ValueError(f"Invalid reason_codes for {dialogue_id}")
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("record_sha256"), str)
            or not _SHA256.fullmatch(source["record_sha256"])
        ):
            raise ValueError(f"Invalid source.record_sha256 for {dialogue_id}")
        seen.add(dialogue_id)
        queue.append(row)

    canonical_lines = corpus_data.splitlines(keepends=True)
    canonical = {}
    for record in read_input_records(corpus):
        if isinstance(record, InputRecordFailure):
            raise TypeError(
                f"Invalid canonical corpus row at {corpus}:{record.line_number}: {record.message}"
            )
        if record.dialogue_id in seen:
            if record.dialogue_id in canonical:
                raise ValueError(f"Duplicate canonical dialogue: {record.dialogue_id}")
            canonical[record.dialogue_id] = (
                record,
                canonical_lines[record.line_number - 1],
            )

    frozen = {}
    for row, line in _jsonl_rows(assignments, assignments_data):
        dialogue_id = row.get("dialogue_id")
        if dialogue_id in seen:
            if dialogue_id in frozen:
                raise ValueError(f"Duplicate frozen assignment: {dialogue_id}")
            frozen[dialogue_id] = line

    ids = [row["dialogue_id"] for row in queue]
    if set(canonical) != seen:
        raise ValueError(f"Missing canonical dialogue: {sorted(seen - set(canonical))}")
    if set(frozen) != seen:
        raise ValueError(f"Missing frozen assignment: {sorted(seen - set(frozen))}")
    for row in queue:
        dialogue_id = row["dialogue_id"]
        if canonical[dialogue_id][0].record_sha256 != row["source"]["record_sha256"]:
            raise ValueError(f"Canonical record SHA mismatch: {dialogue_id}")

    input_lines = [_line_bytes(canonical[dialogue_id][1]) for dialogue_id in ids]
    assignment_lines = [_line_bytes(frozen[dialogue_id]) for dialogue_id in ids]
    if len(input_lines) != len(ids) or len(assignment_lines) != len(ids):
        raise ValueError("Retry input and assignment row counts differ from queue")
    input_data = b"".join(input_lines)
    assignment_data = b"".join(assignment_lines)
    selected_input = [row for row, _ in _jsonl_rows(output / "input.jsonl", input_data)]
    selected_assignments = [
        row for row, _ in _jsonl_rows(output / "assignments.jsonl", assignment_data)
    ]
    if (
        [row.get("dialogue_id") for row in selected_input] != ids
        or [row.get("dialogue_id") for row in selected_assignments] != ids
        or any(
            canonical_json_sha256(row) != queue[index]["source"]["record_sha256"]
            for index, row in enumerate(selected_input)
        )
    ):
        raise ValueError("Prepared retry rows do not match queue order and provenance")
    manifest = {
        "schema_version": "1.0",
        "attempt": attempt,
        "source_retry_queue": str(rejects),
        "source_retry_queue_sha256": _sha256(rejects_data),
        "canonical_corpus": str(corpus),
        "canonical_corpus_sha256": _sha256(corpus_data),
        "frozen_assignments": str(assignments),
        "frozen_assignments_sha256": _sha256(assignments_data),
        "input_jsonl_sha256": _sha256(input_data),
        "assignments_jsonl_sha256": _sha256(assignment_data),
        "dialogue_count": len(ids),
        "dialogues": [
            {
                "dialogue_id": row["dialogue_id"],
                "previous_render_fingerprint": row["render_fingerprint"],
                "reason_codes": row["reason_codes"],
                "record_sha256": row["source"]["record_sha256"],
            }
            for row in queue
        ],
    }
    payloads = (
        input_data,
        assignment_data,
        (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    output.mkdir(parents=True, exist_ok=True)
    staged = []
    published = []
    try:
        for data in payloads:
            staged.append(_stage(output, data))
        for temporary, path in zip(staged, paths, strict=True):
            os.link(temporary, path)
            published.append(path)
    except BaseException:
        for path in published:
            path.unlink()
        raise
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rejects", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--assignments", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempt", required=True, type=int)
    args = parser.parse_args()
    manifest = build_retry_batch(
        args.rejects, args.corpus, args.assignments, args.output, args.attempt
    )
    print(
        f"Built retry batch with {manifest['dialogue_count']} dialogues in {args.output}"
    )


if __name__ == "__main__":
    main()
