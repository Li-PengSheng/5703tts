from __future__ import annotations

from pathlib import Path

import pytest

from tts5703.input.records import (
    InputRecord,
    InputRecordFailure,
    canonical_json_sha256,
    discover_input_paths,
    read_input_records,
)


def _records(path: Path) -> list[InputRecord]:
    return [item for item in read_input_records(path) if isinstance(item, InputRecord)]


def _failures(path: Path) -> list[InputRecordFailure]:
    return [
        item
        for item in read_input_records(path)
        if isinstance(item, InputRecordFailure)
    ]


def test_single_json_record(tmp_path: Path) -> None:
    path = tmp_path / "one.json"
    path.write_text('{"dialogue_id":"D1","turns":[]}', encoding="utf-8")

    record = _records(path)[0]
    assert record.container_path == path
    assert record.source_format == "json"
    assert record.line_number is None
    assert record.raw == {"dialogue_id": "D1", "turns": []}
    assert record.dialogue_id == "D1"
    assert record.record_sha256 == canonical_json_sha256(record.raw)


def test_record_raw_is_an_isolated_deep_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "one.json"
    path.write_text('{"dialogue_id":"D1","x":{"nested":[1,2]}}', encoding="utf-8")
    record = _records(path)[0]
    expected = record.raw
    expected_hash = record.record_sha256

    mutated = record.raw
    mutated["new"] = True
    mutated["x"]["nested"].append(3)

    assert record.raw != mutated
    assert record.raw == expected
    assert record.record_sha256 == expected_hash
    assert record.record_sha256 == canonical_json_sha256(record.raw)


def test_jsonl_skips_blanks_and_continues_after_bad_records(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        '\n{"dialogue_id":"D1","turns":[]}\nnot-json\n'
        '[1,2]\n{"dialogue_id":"D2","turns":[]}\n',
        encoding="utf-8",
    )

    results = read_input_records(path)
    records = [item for item in results if isinstance(item, InputRecord)]
    failures = [item for item in results if isinstance(item, InputRecordFailure)]
    assert [(item.dialogue_id, item.line_number) for item in records] == [
        ("D1", 2),
        ("D2", 5),
    ]
    assert [item.line_number for item in failures] == [3, 4]
    assert "JSON object" in failures[1].message


def test_duplicate_keys_and_missing_dialogue_ids_fail_per_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"dialogue_id":"D1","nested":{"x":1,"x":2}}\n'
        '{"turns":[]}\n{"dialogue_id":"  ","turns":[]}\n'
        '{"dialogue_id":"D2","turns":[]}\n',
        encoding="utf-8",
    )

    assert [item.dialogue_id for item in _records(path)] == ["D2"]
    failures = _failures(path)
    assert [item.line_number for item in failures] == [1, 2, 3]
    assert "duplicate object key" in failures[0].message
    assert all("dialogue_id" in item.message for item in failures[1:])


def test_record_hash_uses_canonical_parsed_json(tmp_path: Path) -> None:
    compact = tmp_path / "compact.json"
    reordered = tmp_path / "reordered.json"
    compact.write_text('{"dialogue_id":"D1","x":{"a":1,"b":2}}', encoding="utf-8")
    reordered.write_text(
        '{\n  "x": {"b": 2, "a": 1},\n  "dialogue_id": "D1"\n}\n',
        encoding="utf-8",
    )

    assert _records(compact)[0].record_sha256 == _records(reordered)[0].record_sha256


def test_json_and_jsonl_use_the_same_canonical_record_hash(tmp_path: Path) -> None:
    json_path = tmp_path / "record.json"
    jsonl_path = tmp_path / "record.jsonl"
    json_path.write_text(
        '{"dialogue_id":"D1","nested":{"a":1,"b":2}}', encoding="utf-8"
    )
    jsonl_path.write_text(
        '{"nested":{"b":2,"a":1},"dialogue_id":"D1"}\n', encoding="utf-8"
    )

    assert _records(json_path)[0].record_sha256 == _records(jsonl_path)[0].record_sha256


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_rejects_non_json_constants(tmp_path: Path, constant: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(f'{{"dialogue_id":"D1","value":{constant}}}', encoding="utf-8")

    failures = _failures(path)
    assert len(failures) == 1
    assert "invalid JSON constant" in failures[0].message


def test_jsonl_rejects_non_json_constants_and_continues(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"dialogue_id":"bad-nan","value":NaN}\n'
        '{"dialogue_id":"bad-positive","value":Infinity}\n'
        '{"dialogue_id":"bad-negative","value":-Infinity}\n'
        '{"dialogue_id":"good","value":1}\n',
        encoding="utf-8",
    )

    assert [failure.line_number for failure in _failures(path)] == [1, 2, 3]
    assert [record.dialogue_id for record in _records(path)] == ["good"]


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_read_oserror_becomes_input_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    path = tmp_path / f"record{suffix}"
    path.write_text('{"dialogue_id":"D1"}\n', encoding="utf-8")
    original_read_text = Path.read_text

    def fail_target(self: Path, *args: object, **kwargs: object) -> str:
        if self == path:
            raise PermissionError("synthetic permission error")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_target)

    results = read_input_records(path)
    assert len(results) == 1
    assert isinstance(results[0], InputRecordFailure)
    assert results[0].line_number is None
    assert "synthetic permission error" in results[0].message


def test_line_moves_do_not_change_identity_and_edit_is_record_local(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        '{"dialogue_id":"A","value":1}\n{"dialogue_id":"B","value":2}\n',
        encoding="utf-8",
    )
    second.write_text(
        '\n{"dialogue_id":"B","value":3}\n\n{"dialogue_id":"A","value":1}\n',
        encoding="utf-8",
    )

    first_hashes = {item.dialogue_id: item.record_sha256 for item in _records(first)}
    second_records = _records(second)
    second_hashes = {item.dialogue_id: item.record_sha256 for item in second_records}
    assert first_hashes["A"] == second_hashes["A"]
    assert first_hashes["B"] != second_hashes["B"]
    assert (
        next(item for item in second_records if item.dialogue_id == "A").line_number
        == 4
    )


def test_directory_discovery_is_deterministic_for_json_and_jsonl(
    tmp_path: Path,
) -> None:
    for name in ("z.jsonl", "nested/m.json", "a.json", "ignored.txt"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    assert [
        path.relative_to(tmp_path).as_posix() for path in discover_input_paths(tmp_path)
    ] == [
        "a.json",
        "nested/m.json",
        "z.jsonl",
    ]
