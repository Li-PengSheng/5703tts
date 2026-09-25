from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tts5703.input.records import canonical_json_sha256

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_production_acceptance_manifest.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_production_acceptance_manifest", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _corpus(tmp_path: Path, ids: list[str]) -> Path:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "".join(
            json.dumps({"dialogue_id": dialogue_id, "text": dialogue_id}) + "\n"
            for dialogue_id in ids
        )
    )
    return path


def _result(
    dialogue_id: str,
    *,
    action: str = "rendered",
    status: str = "success",
    outcome: str | None = "pass",
    codes: list[str] | None = None,
) -> dict:
    record = {"dialogue_id": dialogue_id, "text": dialogue_id}
    result = {
        "dialogue_id": dialogue_id,
        "source": {"record_sha256": canonical_json_sha256(record)},
        "action": action,
        "status": status,
        "render_fingerprint": "a" * 64,
        "output_dir": f"out/{dialogue_id}",
        "error": {"type": "QualityRejected", "message": "rejected"}
        if action == "quality_rejected"
        else None,
    }
    if outcome is not None:
        result["quality"] = {
            "outcome": outcome,
            "issues": [{"code": code} for code in (codes or [])],
        }
    return result


def _batch(
    tmp_path: Path,
    name: str,
    results: list[dict],
    *,
    backend: str = "higgs",
    version: str = "2.0",
) -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {"manifest_version": version, "backend": backend, "results": results}
        )
    )
    return path


def _build(corpus: Path, *sources: tuple[str, str, Path]) -> dict:
    return builder.build_manifest(corpus, list(sources))


def test_pass_resume_partial_order_and_deterministic_rebuild(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, ["B", "A", "C"])
    first = _batch(
        tmp_path, "first.json", [_result("A"), _result("B", action="resumed")]
    )
    manifest = _build(corpus, ("first_pass", "first", first))
    assert [row["dialogue_id"] for row in manifest["dialogues"]] == ["B", "A", "C"]
    assert [row["final_status"] for row in manifest["dialogues"]] == [
        "accepted_first_pass",
        "accepted_first_pass",
        "not_attempted",
    ]
    assert manifest["summary"]["accepted_total"] == 2
    assert manifest["summary"]["not_attempted"] == 1
    assert manifest["dialogues"][0]["history"][0]["action"] == "resumed"
    assert manifest["dialogues"][0]["accepted_render_fingerprint"] == "a" * 64
    output = tmp_path / "acceptance.json"
    assert (
        builder.main(
            [
                "--corpus",
                str(corpus),
                "--first-pass",
                f"first={first}",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    before = output.read_bytes()
    assert (
        builder.main(
            [
                "--corpus",
                str(corpus),
                "--first-pass",
                f"first={first}",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.read_bytes() == before
    assert (
        builder.main(
            [
                "--corpus",
                str(corpus),
                "--first-pass",
                f"first={first}",
                "--output",
                str(output),
                "--require-complete",
            ]
        )
        == 1
    )
    assert output.read_bytes() == before


def test_reject_retry_pass_and_still_rejected(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, ["A", "B"])
    rejected = [
        _result(
            item,
            action="quality_rejected",
            status="failed",
            outcome="reject",
            codes=["abnormal_tail"],
        )
        for item in ("A", "B")
    ]
    first = _batch(tmp_path, "first.json", rejected)
    retry = _batch(tmp_path, "retry.json", [_result("A"), rejected[1]])
    manifest = _build(
        corpus, ("first_pass", "first", first), ("retry", "retry1", retry)
    )
    assert [row["final_status"] for row in manifest["dialogues"]] == [
        "accepted_after_retry",
        "quality_rejected_pending",
    ]
    assert manifest["summary"]["accepted_after_retry"] == 1
    assert manifest["summary"]["quality_rejected_pending"] == 1
    assert manifest["dialogues"][0]["accepted_source_label"] == "retry1"
    assert manifest["dialogues"][1]["history"][-1]["quality_reason_codes"] == [
        "abnormal_tail"
    ]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            _result("A", action="quality_rejected", status="failed", outcome="pass"),
            "failed_integrity",
        ),
        (_result("A", action="rendered", outcome=None), "failed_integrity"),
        (
            _result("A", action="render_failed", status="failed", outcome=None),
            "failed_render_or_input",
        ),
        (
            _result("A", action="input_error", status="failed", outcome=None),
            "failed_render_or_input",
        ),
        (
            _result(
                "A", action="excluded_known_issue", status="excluded", outcome=None
            ),
            "excluded_known_issue",
        ),
    ],
)
def test_unaccepted_dispositions(tmp_path: Path, result: dict, expected: str) -> None:
    corpus = _corpus(tmp_path, ["A"])
    if expected == "failed_integrity" and result["action"] == "quality_rejected":
        result["quality_observation"] = {
            "outcome": "indeterminate",
            "issues": [{"code": "artifact_integrity_failure"}],
        }
    first = _batch(tmp_path, "first.json", [result])
    assert (
        _build(corpus, ("first_pass", "first", first))["dialogues"][0]["final_status"]
        == expected
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "sha",
        "duplicate",
        "version",
        "missing_version",
        "bad_json",
        "duplicate_key",
    ],
)
def test_bad_batch_fails_closed(tmp_path: Path, mutation: str) -> None:
    corpus = _corpus(tmp_path, ["A"])
    first = _batch(tmp_path, "first.json", [_result("A")])
    data = json.loads(first.read_text())
    if mutation == "unknown":
        data["results"][0]["dialogue_id"] = "Z"
    elif mutation == "sha":
        data["results"][0]["source"]["record_sha256"] = "0" * 64
    elif mutation == "duplicate":
        data["results"].append(data["results"][0])
    elif mutation == "version":
        data["manifest_version"] = "1.0"
    elif mutation == "missing_version":
        del data["manifest_version"]
    elif mutation == "bad_json":
        first.write_text("{")
    elif mutation == "duplicate_key":
        first.write_text(
            first.read_text().replace(
                '"manifest_version":',
                '"manifest_version": "2.0", "manifest_version":',
                1,
            )
        )
    if mutation not in {"bad_json", "duplicate_key"}:
        first.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        _build(corpus, ("first_pass", "first", first))


def test_overlapping_sources_and_labels_fail(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, ["A"])
    first = _batch(tmp_path, "first.json", [_result("A")])
    with pytest.raises(ValueError, match="Overlapping"):
        _build(corpus, ("first_pass", "one", first), ("first_pass", "two", first))
    with pytest.raises(ValueError, match="duplicate source label"):
        _build(corpus, ("first_pass", "one", first), ("retry", "one", first))


def test_retry_of_accepted_or_unattempted_fails(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, ["A", "B"])
    first = _batch(tmp_path, "first.json", [_result("A")])
    for dialogue_id in ("A", "B"):
        retry = _batch(tmp_path, "retry.json", [_result(dialogue_id)])
        with pytest.raises(ValueError, match="Retry lacks prior"):
            _build(corpus, ("first_pass", "first", first), ("retry", "retry", retry))


def test_ordered_retries_and_source_provenance(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, ["A"])
    reject = _result(
        "A",
        action="quality_rejected",
        status="failed",
        outcome="reject",
        codes=["z", "a", "z"],
    )
    first = _batch(tmp_path, "first.json", [reject])
    retry1 = _batch(tmp_path, "retry1.json", [reject])
    retry2 = _batch(tmp_path, "retry2.json", [_result("A")])
    manifest = _build(
        corpus,
        ("first_pass", "first", first),
        ("retry", "retry1", retry1),
        ("retry", "retry2", retry2),
    )
    assert [source["label"] for source in manifest["sources"]] == [
        "first",
        "retry1",
        "retry2",
    ]
    assert [
        attempt["source_label"] for attempt in manifest["dialogues"][0]["history"]
    ] == ["first", "retry1", "retry2"]
    assert manifest["dialogues"][0]["history"][0]["quality_reason_codes"] == ["a", "z"]
    assert manifest["dialogues"][0]["final_status"] == "accepted_after_retry"


def test_invalid_manifest_does_not_publish_partial_output(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, ["A"])
    first = _batch(tmp_path, "first.json", [_result("A")], version="1.0")
    output = tmp_path / "acceptance.json"
    assert (
        builder.main(
            [
                "--corpus",
                str(corpus),
                "--first-pass",
                f"first={first}",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert not output.exists()


def test_canonical_duplicate_or_invalid_row_fails(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, ["A", "A"])
    first = _batch(tmp_path, "first.json", [_result("A")])
    with pytest.raises(ValueError, match="Duplicate canonical"):
        _build(corpus, ("first_pass", "first", first))
    corpus.write_text('{"dialogue_id": "A", "dialogue_id": "A"}\n')
    with pytest.raises(TypeError, match="Invalid canonical"):
        _build(corpus, ("first_pass", "first", first))


def test_real_pilot_shape_and_require_complete(tmp_path: Path) -> None:
    ids = [f"corpus_v1_{index:06d}" for index in range(30)]
    corpus = _corpus(tmp_path, ids)
    rejected = {ids[13], ids[24]}
    first = _batch(
        tmp_path,
        "first.json",
        [
            _result(
                dialogue_id,
                action="quality_rejected",
                status="failed",
                outcome="reject",
                codes=["abnormal_tail"],
            )
            if dialogue_id in rejected
            else _result(dialogue_id)
            for dialogue_id in ids
        ],
    )
    retry = _batch(
        tmp_path,
        "retry.json",
        [_result(dialogue_id) for dialogue_id in (ids[13], ids[24])],
    )
    output = tmp_path / "acceptance.json"
    assert (
        builder.main(
            [
                "--corpus",
                str(corpus),
                "--first-pass",
                f"pilot={first}",
                "--retry",
                f"pilot_retry01={retry}",
                "--output",
                str(output),
                "--require-complete",
            ]
        )
        == 0
    )
    summary = json.loads(output.read_text())["summary"]
    assert {
        key: summary[key]
        for key in (
            "dialogues_total",
            "accepted_total",
            "accepted_first_pass",
            "accepted_after_retry",
            "quality_rejected_pending",
        )
    } == {
        "dialogues_total": 30,
        "accepted_total": 30,
        "accepted_first_pass": 28,
        "accepted_after_retry": 2,
        "quality_rejected_pending": 0,
    }
