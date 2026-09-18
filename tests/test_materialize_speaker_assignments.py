"""Offline tests for final speaker-sidecar materialization."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "materialize_speaker_assignments.py"
SPEC = importlib.util.spec_from_file_location(
    "materialize_speaker_assignments", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
materialize = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = materialize
SPEC.loader.exec_module(materialize)

PREFIX = "You are a helpful assistant.<|endofprompt|>"


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dialogue() -> dict[str, Any]:
    required = {
        "rate": "normal",
        "arousal": 2,
        "affect": "neutral",
        "pause_before": "none",
        "pause_within": 0,
        "hesitations": 0,
    }
    return {
        "schema_version": "1.0",
        "dialogue_id": "dialogue_a",
        "scenario": {
            "speakers": {
                "caller": {"speaker_id": "C123"},
                "counsellor": {"speaker_id": "L456"},
            }
        },
        "turns": [
            {
                "turn_id": "u1",
                "speaker": "User",
                "text": "I need help.",
                "labels": {},
                "acoustic": {"required": required, "best_effort": {}},
            },
            {
                "turn_id": "l1",
                "speaker": "Listener",
                "text": "I am here.",
                "labels": {},
                "acoustic": {"required": required, "best_effort": {}},
            },
        ],
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    source = _write_json(tmp_path / "input" / "dialogue.json", _dialogue())
    speakers = []
    for speaker_id in ("spk_001", "spk_002"):
        cosy_bytes = f"COSY-{speaker_id}".encode()
        higgs_bytes = f"HIGGS-{speaker_id}".encode()
        cosy_path = tmp_path / "refs" / f"{speaker_id}-cosy.wav"
        higgs_path = tmp_path / "refs" / f"{speaker_id}-higgs.wav"
        cosy_path.parent.mkdir(parents=True, exist_ok=True)
        cosy_path.write_bytes(cosy_bytes)
        higgs_path.write_bytes(higgs_bytes)
        speakers.append(
            {
                "speaker_id": speaker_id,
                "source_speaker_id": speaker_id.replace("spk_", "p"),
                "primary_reference": {
                    "prompt_wav": str(cosy_path.relative_to(tmp_path)),
                    "prompt_text": f"prompt for {speaker_id}",
                    "sha256": _sha(cosy_bytes),
                },
                "higgs_reference": {
                    "reference_wav": str(higgs_path.relative_to(tmp_path)),
                    "sha256": _sha(higgs_bytes),
                },
            }
        )
    registry = _write_json(
        tmp_path / "speaker_registry.json",
        {"speaker_pool_version": "test_pool", "speakers": speakers},
    )
    active = _write_json(
        tmp_path / "active_speakers.json",
        {
            "speaker_pool_version": "test_pool",
            "status": "active",
            "selection_basis": "test",
            "active_speaker_ids": ["spk_001", "spk_002"],
            "excluded_from_active_pool": [],
        },
    )
    assignments = tmp_path / "assignments.jsonl"
    assignments.write_text(
        json.dumps(
            {
                "dialogue_id": "dialogue_a",
                "role_assignments": {
                    "caller": "spk_001",
                    "counsellor": "spk_002",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "input": source.parent,
        "source": source,
        "registry": registry,
        "active": active,
        "assignments": assignments,
        "root": tmp_path,
    }


def _run(paths: dict[str, Path], backend: str) -> dict[str, Any]:
    return materialize.materialize_speaker_assignments(
        input_path=paths["input"],
        assignments_path=paths["assignments"],
        registry_path=paths["registry"],
        active_speakers_path=paths["active"],
        manifest_path=paths["root"] / f"sidecar-{backend}.json",
        project_root=paths["root"],
        backend=backend,
    )


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("higgs", {"higgs_reference"}),
        ("cosyvoice", {"cosyvoice_reference"}),
        ("both", {"higgs_reference", "cosyvoice_reference"}),
    ],
)
def test_explicit_backend_emits_only_requested_references(
    tmp_path: Path, backend: str, expected: set[str]
) -> None:
    paths = _fixture(tmp_path)
    original = paths["source"].read_bytes()

    result = _run(paths, backend)

    assert result["backend_target"] == backend
    assert result["source_records_rewritten"] is False
    assert paths["source"].read_bytes() == original
    for role in result["dialogues"][0]["roles"].values():
        assert expected <= set(role)
        assert not ({"higgs_reference", "cosyvoice_reference"} - expected) & set(role)
        assert role["render_speaker_id"].startswith("spk_")


@pytest.mark.parametrize(
    ("backend", "removed_field"),
    [("higgs", "primary_reference"), ("cosyvoice", "higgs_reference")],
)
def test_unselected_reference_is_not_required(
    tmp_path: Path, backend: str, removed_field: str
) -> None:
    paths = _fixture(tmp_path)
    registry = json.loads(paths["registry"].read_text(encoding="utf-8"))
    for speaker in registry["speakers"]:
        speaker.pop(removed_field)
    _write_json(paths["registry"], registry)

    assert _run(paths, backend)["backend_target"] == backend


@pytest.mark.parametrize(
    ("backend", "removed_field", "message"),
    [
        ("higgs", "higgs_reference", "approved higgs_reference"),
        ("cosyvoice", "primary_reference", "primary_reference"),
    ],
)
def test_selected_reference_is_required(
    tmp_path: Path, backend: str, removed_field: str, message: str
) -> None:
    paths = _fixture(tmp_path)
    registry = json.loads(paths["registry"].read_text(encoding="utf-8"))
    registry["speakers"][0].pop(removed_field)
    _write_json(paths["registry"], registry)

    with pytest.raises(materialize.SpeakerMaterializationError, match=message):
        _run(paths, backend)


@pytest.mark.parametrize("backend", ["higgs", "cosyvoice"])
def test_selected_reference_sha_mismatch_fails(tmp_path: Path, backend: str) -> None:
    paths = _fixture(tmp_path)
    registry = json.loads(paths["registry"].read_text(encoding="utf-8"))
    reference = (
        registry["speakers"][0]["higgs_reference"]["reference_wav"]
        if backend == "higgs"
        else registry["speakers"][0]["primary_reference"]["prompt_wav"]
    )
    (tmp_path / reference).write_bytes(b"mutated")

    with pytest.raises(
        materialize.SpeakerMaterializationError, match="SHA-256 mismatch"
    ):
        _run(paths, backend)


def test_cosyvoice_preserves_and_formats_registry_prompt_text(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    result = _run(paths, "cosyvoice")
    reference = result["dialogues"][0]["roles"]["caller"]["cosyvoice_reference"]

    assert reference["prompt_text"] == f"{PREFIX}prompt for spk_001"


@pytest.mark.parametrize("backend", ["", "auto", "unknown"])
def test_backend_target_must_be_explicit_and_supported(
    tmp_path: Path, backend: str
) -> None:
    paths = _fixture(tmp_path)

    with pytest.raises(materialize.SpeakerMaterializationError, match="backend must"):
        _run(paths, backend)


def test_cli_writes_sidecar(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest = tmp_path / "cli-sidecar.json"

    assert (
        materialize.main(
            [
                "--input",
                str(paths["input"]),
                "--assignments",
                str(paths["assignments"]),
                "--registry",
                str(paths["registry"]),
                "--active-speakers",
                str(paths["active"]),
                "--manifest",
                str(manifest),
                "--project-root",
                str(paths["root"]),
                "--backend",
                "both",
            ]
        )
        == 0
    )
    assert json.loads(manifest.read_text(encoding="utf-8"))["backend_target"] == "both"
