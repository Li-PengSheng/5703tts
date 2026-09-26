"""Offline checks for canonical-order production chunk preparation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_production_chunks",
    Path(__file__).resolve().parents[1] / "scripts/build_production_chunks.py",
)
assert _SPEC is not None and _SPEC.loader is not None
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)


def _line(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False) + "\n").encode()


def _record(dialogue_id: str, turns: int = 2) -> dict:
    return {"dialogue_id": dialogue_id, "turns": [{"turn_id": i} for i in range(turns)]}


def _assignment(dialogue_id: str) -> dict:
    return {
        "dialogue_id": dialogue_id,
        "speaker_pool_version": "vctk_v0.2",
        "assignment_policy_version": "speaker_assignment_policy_v0.2",
        "role_assignments": {"caller": "spk_001", "counsellor": "spk_006"},
    }


def _sources(tmp_path: Path, ids: list[str], *, turns: int = 2) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus.jsonl"
    assignments = tmp_path / "assignments.jsonl"
    corpus.write_bytes(b"".join(_line(_record(item, turns)) for item in ids))
    assignments.write_bytes(
        b"".join(_line(_assignment(item)) for item in reversed(ids))
    )
    return corpus, assignments


def _ids(path: Path) -> list[str]:
    return [json.loads(line)["dialogue_id"] for line in path.read_bytes().splitlines()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_split_order_partial_bytes_hashes_and_determinism(tmp_path: Path) -> None:
    ids = ["id_9", "id_10", "id_2", "z", "a"]
    corpus, assignments = _sources(tmp_path, ids)
    corpus.write_bytes(
        b"".join(
            (json.dumps(_record(item), separators=(",", ":")) + "  \n").encode()
            for item in ids
        )
    )
    assignments.write_bytes(
        b"".join(
            (json.dumps(_assignment(item), separators=(",", ":")) + " \n").encode()
            for item in reversed(ids)
        )
    )
    first = tmp_path / "one"
    second = tmp_path / "two"
    result = builder.build_production_chunks(corpus, assignments, first, 2)
    builder.build_production_chunks(corpus, assignments, second, 2)

    assert result["chunk_size"] == 2
    assert result["chunk_count"] == 3
    assert result["dialogues_total"] == 5
    assert result["turns_total"] == 10
    assert [item["chunk_id"] for item in result["chunks"]] == [
        "chunk_000",
        "chunk_001",
        "chunk_002",
    ]
    assert [item["dialogue_count"] for item in result["chunks"]] == [2, 2, 1]
    assert sum(item["turn_count"] for item in result["chunks"]) == 10
    assert result["canonical_corpus_sha256"] == _sha(corpus)
    assert result["frozen_assignments_sha256"] == _sha(assignments)

    source_input = {
        json.loads(line)["dialogue_id"]: line
        for line in corpus.read_bytes().splitlines(keepends=True)
    }
    source_assignment = {
        json.loads(line)["dialogue_id"]: line
        for line in assignments.read_bytes().splitlines(keepends=True)
    }
    all_ids = []
    for index, entry in enumerate(result["chunks"]):
        chunk = first / entry["chunk_id"]
        detail = json.loads((chunk / "chunk_manifest.json").read_bytes())
        expected = ids[index * 2 : (index + 1) * 2]
        assert detail["dialogue_ids"] == expected
        assert detail["chunk_index"] == index
        assert detail["first_dialogue_id"] == expected[0]
        assert detail["last_dialogue_id"] == expected[-1]
        assert detail["dialogue_count"] == len(expected)
        assert detail["turn_count"] == 2 * len(expected)
        assert _ids(chunk / "input.jsonl") == expected
        assert _ids(chunk / "assignments.jsonl") == expected
        assert (chunk / "input.jsonl").read_bytes() == b"".join(
            source_input[item] for item in expected
        )
        assert (chunk / "assignments.jsonl").read_bytes() == b"".join(
            source_assignment[item] for item in expected
        )
        for name, key in (
            ("input.jsonl", "input_jsonl_sha256"),
            ("assignments.jsonl", "assignments_jsonl_sha256"),
            ("chunk_manifest.json", "chunk_manifest_sha256"),
        ):
            assert entry[key] == _sha(chunk / name)
            if name != "chunk_manifest.json":
                assert detail[key] == entry[key]
        all_ids.extend(expected)
    assert all_ids == ids
    assert len(set(all_ids)) == len(ids)
    assert json.loads((first / "production_manifest.json").read_bytes()) == result
    assert {
        path.relative_to(first): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(second): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    ("corpus_data", "assignment_data", "message"),
    [
        (b"", None, "empty"),
        (b'{"dialogue_id": "a", "turns": [}\n', None, "Invalid canonical"),
        (_line({"dialogue_id": "a"}), None, "Malformed canonical turns"),
        (_line(_record("a")) * 2, None, "Duplicate canonical"),
        (None, b'{"dialogue_id": "a", "role_assignments": }\n', "Invalid assignment"),
        (None, _line({"dialogue_id": "a"}), "Malformed frozen assignment"),
        (None, _line(_assignment("a")) * 2, "Duplicate frozen"),
        (None, b"", "Assignment set mismatch"),
        (None, _line(_assignment("other")), "Assignment set mismatch"),
        (
            None,
            _line(_assignment("a")) + _line(_assignment("other")),
            "Assignment set mismatch",
        ),
        (None, b'{"dialogue_id":"a","dialogue_id":"a"}\n', "duplicate JSON key"),
    ],
)
def test_bad_sources_leave_no_output(
    tmp_path: Path,
    corpus_data: bytes | None,
    assignment_data: bytes | None,
    message: str,
) -> None:
    corpus, assignments = _sources(tmp_path, ["a"])
    if corpus_data is not None:
        corpus.write_bytes(corpus_data)
    if assignment_data is not None:
        assignments.write_bytes(assignment_data)
    output = tmp_path / "output"
    with pytest.raises((ValueError, TypeError), match=message):
        builder.build_production_chunks(corpus, assignments, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".output.staging-*"))


def test_existing_output_is_untouched(tmp_path: Path) -> None:
    corpus, assignments = _sources(tmp_path, ["a"])
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("unchanged")
    with pytest.raises(FileExistsError):
        builder.build_production_chunks(corpus, assignments, output)
    assert marker.read_text() == "unchanged"


def test_invalid_chunk_size_leaves_no_output(tmp_path: Path) -> None:
    corpus, assignments = _sources(tmp_path, ["a"])
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="positive integer"):
        builder.build_production_chunks(corpus, assignments, output, 0)
    assert not output.exists()


def test_publication_failure_cleans_only_own_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, assignments = _sources(tmp_path, ["a"])
    keeper = tmp_path / ".output.staging-operator"
    keeper.mkdir()

    def fail_rename(_source: Path, _target: Path) -> None:
        raise OSError("publication failed")

    monkeypatch.setattr(builder.os, "rename", fail_rename)
    with pytest.raises(OSError, match="publication failed"):
        builder.build_production_chunks(corpus, assignments, tmp_path / "output")
    assert keeper.is_dir()
    assert not (tmp_path / "output").exists()
    assert list(tmp_path.glob(".output.staging-*")) == [keeper]


def test_production_shaped_thousand_dialogues(tmp_path: Path) -> None:
    ids = [f"synthetic_{index:04d}" for index in range(1000)]
    corpus, assignments = _sources(tmp_path, ids, turns=3)
    output = tmp_path / "output"
    result = builder.build_production_chunks(corpus, assignments, output, 100)
    assert result["chunk_count"] == 10
    assert result["dialogues_total"] == 1000
    assert result["turns_total"] == 3000
    assert [item["chunk_id"] for item in result["chunks"]] == [
        f"chunk_{index:03d}" for index in range(10)
    ]
    assert all(item["dialogue_count"] == 100 for item in result["chunks"])
    assert sum(item["turn_count"] for item in result["chunks"]) == 3000
    input_ids = []
    assignment_ids = []
    for index in range(10):
        chunk = output / f"chunk_{index:03d}"
        input_ids.extend(_ids(chunk / "input.jsonl"))
        assignment_ids.extend(_ids(chunk / "assignments.jsonl"))
    assert input_ids[:100] == ids[:100]
    assert input_ids[900:] == ids[900:]
    assert input_ids == assignment_ids == ids
    assert len(set(input_ids)) == 1000
