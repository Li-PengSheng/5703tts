"""Select canonical-order production chunks and frozen speaker assignments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from tts5703.input.records import InputRecordFailure, read_input_records


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _assignment_rows(path: Path, data: bytes) -> dict[str, bytes]:
    if path.suffix.lower() != ".jsonl":
        raise ValueError(f"Expected a JSONL assignment file: {path}")
    rows = {}
    for number, line in enumerate(data.splitlines(keepends=True), start=1):
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
                f"Invalid assignment at {path}:{number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise TypeError(f"Assignment at {path}:{number} must be an object")
        dialogue_id = row.get("dialogue_id")
        roles = row.get("role_assignments")
        if (
            not isinstance(dialogue_id, str)
            or not dialogue_id.strip()
            or any(
                not isinstance(row.get(key), str) or not row[key].strip()
                for key in ("speaker_pool_version", "assignment_policy_version")
            )
            or not isinstance(roles, dict)
            or set(roles) != {"caller", "counsellor"}
            or any(
                not isinstance(value, str) or not value.strip()
                for value in roles.values()
            )
            or roles["caller"] == roles["counsellor"]
        ):
            raise ValueError(f"Malformed frozen assignment at {path}:{number}")
        if dialogue_id in rows:
            raise ValueError(f"Duplicate frozen assignment: {dialogue_id}")
        rows[dialogue_id] = line if line.endswith(b"\n") else line + b"\n"
    return rows


def build_production_chunks(
    corpus: Path, assignments: Path, output: Path, chunk_size: int = 100
) -> dict[str, Any]:
    if chunk_size < 1:
        raise ValueError("--chunk-size must be a positive integer")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite production output: {output}")
    if corpus.suffix.lower() != ".jsonl":
        raise ValueError(f"Expected a JSONL corpus: {corpus}")

    corpus_data = corpus.read_bytes()
    assignment_data = assignments.read_bytes()
    source_lines = corpus_data.splitlines(keepends=True)
    records = []
    seen = set()
    for record in read_input_records(corpus):
        if isinstance(record, InputRecordFailure):
            raise TypeError(
                f"Invalid canonical record at {corpus}:{record.line_number}: {record.message}"
            )
        if record.dialogue_id in seen:
            raise ValueError(f"Duplicate canonical dialogue: {record.dialogue_id}")
        turns = record.raw.get("turns")
        if (
            not isinstance(turns, list)
            or not turns
            or any(not isinstance(turn, dict) for turn in turns)
        ):
            raise ValueError(f"Malformed canonical turns: {record.dialogue_id}")
        seen.add(record.dialogue_id)
        line = source_lines[record.line_number - 1]
        records.append(
            (
                record.dialogue_id,
                len(turns),
                line if line.endswith(b"\n") else line + b"\n",
            )
        )
    if corpus.read_bytes() != corpus_data:
        raise ValueError("Canonical corpus changed during validation")
    if not records:
        raise ValueError("Canonical corpus is empty")

    frozen = _assignment_rows(assignments, assignment_data)
    if set(frozen) != seen:
        raise ValueError(
            f"Assignment set mismatch: missing={sorted(seen - set(frozen))}, "
            f"unknown={sorted(set(frozen) - seen)}"
        )

    source = {
        "canonical_corpus": str(corpus),
        "canonical_corpus_sha256": _sha256(corpus_data),
        "frozen_assignments": str(assignments),
        "frozen_assignments_sha256": _sha256(assignment_data),
    }
    manifest = {
        "schema_version": "1.0",
        "chunk_size": chunk_size,
        "dialogues_total": len(records),
        "turns_total": sum(item[1] for item in records),
        "chunk_count": (len(records) + chunk_size - 1) // chunk_size,
        **source,
        "chunks": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        generated_ids = []
        for index, start in enumerate(range(0, len(records), chunk_size)):
            selected = records[start : start + chunk_size]
            ids = [item[0] for item in selected]
            chunk_id = f"chunk_{index:03d}"
            chunk = stage / chunk_id
            chunk.mkdir()
            input_data = b"".join(item[2] for item in selected)
            selected_assignments = b"".join(frozen[dialogue_id] for dialogue_id in ids)
            (chunk / "input.jsonl").write_bytes(input_data)
            (chunk / "assignments.jsonl").write_bytes(selected_assignments)
            detail = {
                "schema_version": "1.0",
                "chunk_id": chunk_id,
                "chunk_index": index,
                "chunk_size": chunk_size,
                "dialogue_count": len(ids),
                "turn_count": sum(item[1] for item in selected),
                "first_dialogue_id": ids[0],
                "last_dialogue_id": ids[-1],
                "dialogue_ids": ids,
                **source,
                "input_jsonl": f"{chunk_id}/input.jsonl",
                "input_jsonl_sha256": _sha256(input_data),
                "assignments_jsonl": f"{chunk_id}/assignments.jsonl",
                "assignments_jsonl_sha256": _sha256(selected_assignments),
            }
            detail_data = _json_bytes(detail)
            (chunk / "chunk_manifest.json").write_bytes(detail_data)
            manifest["chunks"].append(
                {
                    "chunk_id": chunk_id,
                    "dialogue_count": len(ids),
                    "turn_count": detail["turn_count"],
                    "first_dialogue_id": ids[0],
                    "last_dialogue_id": ids[-1],
                    "input_jsonl_sha256": detail["input_jsonl_sha256"],
                    "assignments_jsonl_sha256": detail["assignments_jsonl_sha256"],
                    "chunk_manifest_sha256": _sha256(detail_data),
                }
            )
            if (
                (chunk / "input.jsonl").read_bytes() != input_data
                or (chunk / "assignments.jsonl").read_bytes() != selected_assignments
                or _sha256((chunk / "chunk_manifest.json").read_bytes())
                != manifest["chunks"][-1]["chunk_manifest_sha256"]
                or list(
                    _assignment_rows(chunk / "assignments.jsonl", selected_assignments)
                )
                != ids
            ):
                raise ValueError(f"Generated chunk verification failed: {chunk_id}")
            generated_ids.extend(ids)
        if (
            generated_ids != [item[0] for item in records]
            or len(set(generated_ids)) != len(records)
            or len(manifest["chunks"]) != manifest["chunk_count"]
            or sum(item["dialogue_count"] for item in manifest["chunks"])
            != manifest["dialogues_total"]
            or sum(item["turn_count"] for item in manifest["chunks"])
            != manifest["turns_total"]
        ):
            raise ValueError("Generated partition verification failed")
        top_data = _json_bytes(manifest)
        (stage / "production_manifest.json").write_bytes(top_data)
        if (stage / "production_manifest.json").read_bytes() != top_data:
            raise ValueError("Generated production manifest verification failed")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Refusing to overwrite production output: {output}")
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--assignments", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()
    manifest = build_production_chunks(
        args.corpus, args.assignments, args.output, args.chunk_size
    )
    print(f"Built {manifest['chunk_count']} chunks in {args.output}")


if __name__ == "__main__":
    main()
