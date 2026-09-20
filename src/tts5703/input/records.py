"""Deterministic JSON and JSONL input-record loading."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

SourceFormat = Literal["json", "jsonl"]


class InputRecordError(ValueError):
    """Raised when an input path cannot provide dialogue records."""


class _DuplicateKeyError(ValueError):
    pass


@dataclass(frozen=True, init=False)
class InputRecord:
    container_path: Path
    source_format: SourceFormat
    line_number: int | None
    _raw: dict[str, Any] = field(repr=False)
    dialogue_id: str
    record_sha256: str

    def __init__(
        self,
        *,
        container_path: Path,
        source_format: SourceFormat,
        line_number: int | None,
        raw: dict[str, Any],
        dialogue_id: str,
    ) -> None:
        snapshot = deepcopy(raw)
        object.__setattr__(self, "container_path", container_path)
        object.__setattr__(self, "source_format", source_format)
        object.__setattr__(self, "line_number", line_number)
        object.__setattr__(self, "_raw", snapshot)
        object.__setattr__(self, "dialogue_id", dialogue_id)
        object.__setattr__(self, "record_sha256", canonical_json_sha256(snapshot))

    @property
    def raw(self) -> dict[str, Any]:
        return deepcopy(self._raw)


@dataclass(frozen=True)
class InputRecordFailure:
    container_path: Path
    source_format: SourceFormat
    line_number: int | None
    message: str


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def discover_input_paths(input_path: Path) -> list[Path]:
    """Return supported containers in stable path order."""
    if input_path.is_file():
        if input_path.suffix.lower() not in {".json", ".jsonl"}:
            raise InputRecordError(f"Unsupported input file type: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise InputRecordError(f"Input path does not exist: {input_path}")
    paths = sorted(
        (
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
        ),
        key=lambda path: path.as_posix(),
    )
    if not paths:
        raise InputRecordError(f"No JSON or JSONL files found under {input_path}")
    return paths


def _parse_record(
    text: str,
    *,
    container_path: Path,
    source_format: SourceFormat,
    line_number: int | None,
) -> InputRecord | InputRecordFailure:
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_non_json_constant,
        )
    except ValueError as error:
        return InputRecordFailure(
            container_path, source_format, line_number, str(error)
        )
    if not isinstance(raw, dict):
        return InputRecordFailure(
            container_path,
            source_format,
            line_number,
            "record must be a JSON object",
        )
    dialogue_id = raw.get("dialogue_id")
    if not isinstance(dialogue_id, str) or not dialogue_id.strip():
        return InputRecordFailure(
            container_path,
            source_format,
            line_number,
            "record is missing a nonblank dialogue_id",
        )
    return InputRecord(
        container_path=container_path,
        source_format=source_format,
        line_number=line_number,
        raw=raw,
        dialogue_id=dialogue_id,
    )


def read_input_records(
    input_path: Path,
) -> list[InputRecord | InputRecordFailure]:
    """Read every record, retaining per-record failures so later lines survive."""
    results: list[InputRecord | InputRecordFailure] = []
    for path in discover_input_paths(input_path):
        source_format: SourceFormat = (
            "jsonl" if path.suffix.lower() == ".jsonl" else "json"
        )
        if source_format == "json":
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as error:
                results.append(
                    InputRecordFailure(path, source_format, None, str(error))
                )
            else:
                results.append(
                    _parse_record(
                        text,
                        container_path=path,
                        source_format=source_format,
                        line_number=None,
                    )
                )
            continue

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            results.append(InputRecordFailure(path, source_format, None, str(error)))
            continue
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            results.append(
                _parse_record(
                    line,
                    container_path=path,
                    source_format=source_format,
                    line_number=line_number,
                )
            )
    return results
