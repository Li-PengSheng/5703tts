from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from tts5703.input.records import canonical_json_sha256, read_input_records

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_quality_retry_batch", ROOT / "scripts" / "build_quality_retry_batch.py"
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _record(dialogue_id: str) -> dict:
    return {"dialogue_id": dialogue_id, "text": f"Canonical dialogue {dialogue_id}"}


def _queue_row(record: dict) -> dict:
    return {
        "dialogue_id": record["dialogue_id"],
        "source": {"record_sha256": canonical_json_sha256(record)},
        "render_fingerprint": "a" * 64,
        "reason_codes": ["abnormal_tail"],
        "output_dir": "previous/output",
        "quality_sidecar": "previous/output/quality.json",
    }


def _seed(
    tmp_path: Path, order: tuple[str, ...] = ("A",)
) -> tuple[Path, Path, Path, Path]:
    rejects = tmp_path / "quality_rejected.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    assignments = tmp_path / "speaker_assignments.jsonl"
    output = tmp_path / "retry"
    rejects.write_text(
        "".join(json.dumps(_queue_row(_record(item))) + "\n" for item in order)
    )
    corpus.write_text(
        "".join(json.dumps(_record(item)) + "\n" for item in sorted(set(order)))
    )
    assignments.write_text(
        "".join(
            json.dumps(
                {"dialogue_id": item, "role_assignments": {"caller": f"spk_{item}"}}
            )
            + "\n"
            for item in sorted(set(order))
        )
    )
    return rejects, corpus, assignments, output


def _build(paths: tuple[Path, Path, Path, Path]) -> dict:
    return builder.build_retry_batch(*paths, 1)


def test_one_row_copies_canonical_and_frozen_bytes_with_manifest_hashes(
    tmp_path: Path,
) -> None:
    paths = _seed(tmp_path)
    rejects, corpus, assignments, output = paths
    canonical_line = corpus.read_bytes()
    frozen_line = assignments.read_bytes()
    manifest = _build(paths)

    assert (output / "input.jsonl").read_bytes() == canonical_line
    assert (output / "assignments.jsonl").read_bytes() == frozen_line
    assert manifest == json.loads((output / "retry_manifest.json").read_text())
    assert manifest["dialogue_count"] == 1
    assert manifest["dialogues"] == [
        {
            "dialogue_id": "A",
            "previous_render_fingerprint": "a" * 64,
            "reason_codes": ["abnormal_tail"],
            "record_sha256": canonical_json_sha256(_record("A")),
        }
    ]
    assert manifest["attempt"] == 1
    assert manifest["schema_version"] == "1.0"
    for key, path in (
        ("source_retry_queue_sha256", rejects),
        ("canonical_corpus_sha256", corpus),
        ("frozen_assignments_sha256", assignments),
        ("input_jsonl_sha256", output / "input.jsonl"),
        ("assignments_jsonl_sha256", output / "assignments.jsonl"),
    ):
        assert manifest[key] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "Canonical dialogue" not in (output / "retry_manifest.json").read_text()


def test_retry_queue_order_controls_both_outputs(tmp_path: Path) -> None:
    paths = _seed(tmp_path, ("B", "A"))
    manifest = _build(paths)
    output = paths[3]
    for name in ("input.jsonl", "assignments.jsonl"):
        assert [
            json.loads(line)["dialogue_id"]
            for line in (output / name).read_text().splitlines()
        ] == ["B", "A"]
    assert [item["dialogue_id"] for item in manifest["dialogues"]] == ["B", "A"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_queue", "Duplicate retry dialogue_id"),
        ("missing_canonical", "Missing canonical dialogue"),
        ("duplicate_canonical", "Duplicate canonical dialogue"),
        ("sha_mismatch", "Canonical record SHA mismatch"),
        ("missing_assignment", "Missing frozen assignment"),
        ("duplicate_assignment", "Duplicate frozen assignment"),
    ],
)
def test_invalid_sources_fail_before_publication(
    tmp_path: Path, mutation: str, message: str
) -> None:
    paths = _seed(tmp_path)
    rejects, corpus, assignments, output = paths
    if mutation == "duplicate_queue":
        rejects.write_bytes(rejects.read_bytes() * 2)
    elif mutation == "missing_canonical":
        corpus.write_text(json.dumps(_record("B")) + "\n")
    elif mutation == "duplicate_canonical":
        corpus.write_bytes(corpus.read_bytes() * 2)
    elif mutation == "sha_mismatch":
        corpus.write_text(json.dumps({**_record("A"), "text": "changed"}) + "\n")
    elif mutation == "missing_assignment":
        assignments.write_text(json.dumps({"dialogue_id": "B"}) + "\n")
    else:
        assignments.write_bytes(assignments.read_bytes() * 2)

    with pytest.raises(ValueError, match=message):
        _build(paths)
    assert not any((output / name).exists() for name in builder._OUTPUT_NAMES)


def test_empty_queue_builds_empty_pack(tmp_path: Path) -> None:
    paths = _seed(tmp_path, ())
    manifest = _build(paths)
    assert (paths[3] / "input.jsonl").read_bytes() == b""
    assert (paths[3] / "assignments.jsonl").read_bytes() == b""
    assert manifest["dialogue_count"] == 0
    assert manifest["dialogues"] == []


@pytest.mark.parametrize("name", builder._OUTPUT_NAMES)
def test_existing_retry_artifact_is_never_overwritten(
    tmp_path: Path, name: str
) -> None:
    paths = _seed(tmp_path)
    output = paths[3]
    output.mkdir()
    existing = output / name
    existing.write_bytes(b"existing attempt")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _build(paths)
    assert existing.read_bytes() == b"existing attempt"
    assert [path.name for path in output.iterdir()] == [name]


def test_committed_frozen_assignment_row_is_copied_unchanged(tmp_path: Path) -> None:
    corpus = ROOT / "data/final/corpus_v1_1000.jsonl"
    assignments = ROOT / "data/speaker_assignment/v0.2/speaker_assignments.jsonl"
    record = read_input_records(corpus)[0]
    assert not isinstance(record, builder.InputRecordFailure)
    rejects = tmp_path / "quality_rejected.jsonl"
    rejects.write_text(json.dumps(_queue_row(record.raw)) + "\n")
    output = tmp_path / "retry"
    builder.build_retry_batch(rejects, corpus, assignments, output, 1)

    expected = next(
        line
        for line in assignments.read_bytes().splitlines(keepends=True)
        if json.loads(line)["dialogue_id"] == record.dialogue_id
    )
    assert (output / "assignments.jsonl").read_bytes() == expected


@pytest.mark.parametrize(
    "row",
    [
        {"dialogue_id": " ", "source": {"record_sha256": "a" * 64}},
        {"dialogue_id": "A", "source": {"record_sha256": "bad"}},
        {"dialogue_id": "A", "source": {"record_sha256": "a" * 64}, "reason_codes": []},
        {
            "dialogue_id": "A",
            "source": {"record_sha256": "a" * 64},
            "render_fingerprint": "bad",
        },
    ],
)
def test_invalid_queue_fields_fail_closed(tmp_path: Path, row: dict) -> None:
    paths = _seed(tmp_path)
    queue_row = _queue_row(_record("A"))
    queue_row.update(row)
    paths[0].write_text(json.dumps(queue_row) + "\n")
    with pytest.raises(ValueError):
        _build(paths)


def test_queue_requires_jsonl_objects(tmp_path: Path) -> None:
    paths = _seed(tmp_path)
    paths[0].write_text("[]\n")
    with pytest.raises(TypeError, match="Expected JSON object"):
        _build(paths)
