from __future__ import annotations

import asyncio
import datetime
import hashlib
import inspect
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydub import AudioSegment

from tts5703 import batch, batch_identity, cli
from tts5703.pipeline import PipelineResult


def _record(dialogue_id: str, text: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "dialogue_id": dialogue_id,
        "scenario": {
            "speakers": {
                "caller": {"speaker_id": f"C-{dialogue_id}"},
                "counsellor": {"speaker_id": f"L-{dialogue_id}"},
            }
        },
        "turns": [
            {
                "turn_id": f"{dialogue_id}-turn",
                "speaker": "User",
                "text": text or f"Dialogue {dialogue_id}",
                "labels": {"opaque": [dialogue_id]},
                "acoustic": {
                    "required": {
                        "rate": "normal",
                        "arousal": 2,
                        "affect": "neutral",
                        "pause_before": "none",
                        "pause_within": 0,
                        "hesitations": 0,
                    },
                    "best_effort": {},
                },
            }
        ],
    }


def _record_with_turn_count(dialogue_id: str, count: int) -> dict[str, Any]:
    record = _record(dialogue_id)
    template = record["turns"][0]
    record["turns"] = []
    for ordinal in range(1, count + 1):
        turn = deepcopy(template)
        turn["turn_id"] = f"{dialogue_id}-turn-{ordinal}"
        turn["text"] = f"Dialogue {dialogue_id}, turn {ordinal}"
        record["turns"].append(turn)
    return record


def _reference_bytes(path: str) -> bytes:
    return f"reference:{path}".encode()


def _reference_sha(path: str) -> str:
    return hashlib.sha256(_reference_bytes(path)).hexdigest()


def _role(
    upstream: str,
    speaker: str,
    reference_stem: str,
    higgs_digest: str | None = None,
) -> dict:
    higgs_path = f"{reference_stem}-higgs.wav"
    cosy_path = f"{reference_stem}-cosy.wav"
    return {
        "upstream_role": upstream,
        "upstream_scenario_speaker_id": f"scenario-{speaker}",
        "render_speaker_id": speaker,
        "higgs_reference": {
            "reference_wav": higgs_path,
            "sha256": higgs_digest or _reference_sha(higgs_path),
        },
        "cosyvoice_reference": {
            "prompt_wav": cosy_path,
            "prompt_text": f"Prompt for {speaker}",
            "sha256": _reference_sha(cosy_path),
        },
    }


def _sidecar(
    dialogue_ids: tuple[str, ...],
    *,
    b_speaker: str = "spk_B_caller",
    b_reference_sha: str | None = None,
    registry_sha: str = "a" * 64,
    active_sha: str = "b" * 64,
) -> dict[str, Any]:
    entries = []
    for dialogue_id in dialogue_ids:
        caller = b_speaker if dialogue_id == "B" else f"spk_{dialogue_id}_caller"
        caller_sha = b_reference_sha if dialogue_id == "B" else None
        entries.append(
            {
                "dialogue_id": dialogue_id,
                "source": {"record_sha256": "provenance-only"},
                "roles": {
                    "caller": _role(
                        "User", caller, f"refs/{dialogue_id}-caller", caller_sha
                    ),
                    "counsellor": _role(
                        "Listener",
                        f"spk_{dialogue_id}_counsellor",
                        f"refs/{dialogue_id}-counsellor",
                    ),
                },
            }
        )
    return {
        "assignment_sha256": "c" * 64,
        "registry_sha256": registry_sha,
        "active_speakers_sha256": active_sha,
        "dialogues": entries,
    }


def _config(engine: str = "higgs") -> dict[str, Any]:
    return {
        "tts": {
            "engine": engine,
            "higgs": {
                "server_executable": "tools/sgl-omni",
                "model_dir": "models/higgs",
                "ffmpeg_bin": "ffmpeg",
            },
            "cosyvoice": {
                "python_bin": "python",
                "repo_dir": "third_party/CosyVoice",
                "model_dir": "models/CosyVoice",
            },
        },
        "fade_ms": 5,
        "telephone": {
            "sample_rate": 8000,
            "channels": 1,
            "high_pass_hz": 300,
            "low_pass_hz": 3400,
            "volume_db_reduction": 2,
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, Any] | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [item if isinstance(item, str) else json.dumps(item) for item in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_success_artifacts(
    output_root: Path, record: Any, config: dict[str, Any]
) -> Path:
    dialogue_id = record.dialogue_id
    out_dir = output_root / dialogue_id
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = batch_identity.backend_identity(config)
    turns = []
    previous_end = 0.0
    for ordinal, source in enumerate(record.raw["turns"], start=1):
        turn_audio = f"turn_{ordinal:03d}.wav"
        AudioSegment.silent(duration=200).export(out_dir / turn_audio, format="wav")
        end_sec = previous_end + 0.2
        turns.append(
            {
                "source_identity": {
                    "ordinal": ordinal,
                    "source_turn_id": source["turn_id"],
                },
                "labels": deepcopy(source["labels"]),
                "requested": {"acoustic": deepcopy(source["acoustic"])},
                "execution": {
                    "synthesis_status": "synthesized",
                    "turn_audio": turn_audio,
                },
                "timing": {"start_sec": previous_end, "end_sec": end_sec},
            }
        )
        previous_end = end_sec
    for suffix in ("clean.wav", "telephone.wav"):
        AudioSegment.silent(duration=max(200, round(previous_end * 1000))).export(
            out_dir / f"{dialogue_id}_{suffix}", format="wav"
        )
    metadata = {
        "dialogue_id": dialogue_id,
        "clean_audio": f"{dialogue_id}_clean.wav",
        "telephone_audio": f"{dialogue_id}_telephone.wav",
        "tts": {
            "engine": config["tts"]["engine"],
            "backend_identity": backend,
            "control_mapping": backend["control_mapping"],
        },
        "provenance": {"record_sha256": record.record_sha256},
        "turns": turns,
    }
    (out_dir / f"{dialogue_id}_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return out_dir


def _change_metadata_value(path: Path, keys: tuple[str | int, ...], value: Any) -> None:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    target = metadata
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path.write_text(json.dumps(metadata), encoding="utf-8")


def _invoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    input_path: Path,
    sidecar: dict[str, Any],
    policy: dict[str, Any] | None = None,
    resume: bool = False,
    engine: str = "higgs",
    config: dict[str, Any] | None = None,
    failures: set[str] | None = None,
) -> tuple[int, dict[str, Any], list[str]]:
    output = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    sidecar_path = tmp_path / "sidecar.json"
    policy_path = tmp_path / "policy.json"
    loaded_config = config or _config(engine)
    config_path.write_text(
        json.dumps(loaded_config, sort_keys=True) + "\n", encoding="utf-8"
    )
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    if policy is not None:
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for entry in sidecar.get("dialogues", []):
        for role in entry.get("roles", {}).values():
            for reference_key, path_key in (
                ("higgs_reference", "reference_wav"),
                ("cosyvoice_reference", "prompt_wav"),
            ):
                reference = role.get(reference_key)
                if not isinstance(reference, dict):
                    continue
                declared = reference.get(path_key)
                if not isinstance(declared, str):
                    continue
                path = Path(declared)
                if not path.is_absolute():
                    path = tmp_path / path
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(_reference_bytes(declared))
    attempted: list[str] = []

    async def fake_run_dialogue(
        record: object,
        sidecar_manifest: dict,
        config: dict,
        output_root: Path,
        *,
        project_root: Path,
    ) -> PipelineResult:
        dialogue_id = record.dialogue_id
        attempted.append(dialogue_id)
        if failures and dialogue_id in failures:
            (output_root / dialogue_id).mkdir(parents=True, exist_ok=True)
            return PipelineResult(
                dialogue_id,
                "failed",
                "synthetic render failure",
                error_type="SyntheticFailure",
            )
        out_dir = _write_success_artifacts(output_root, record, config)
        return PipelineResult(dialogue_id, "success", out_dir=out_dir)

    monkeypatch.setattr(cli, "configure_logging", lambda *_: None)
    monkeypatch.setattr(cli, "load_config", lambda _: deepcopy(loaded_config))
    monkeypatch.setattr(batch, "run_dialogue", fake_run_dialogue)

    async def run_batch_with_distinct_start(**kwargs: Any) -> int:
        # The CLI tests run multiple batches within one second. Give each test
        # attempt a distinct whole-second timestamp without changing production.
        kwargs["run_started"] = datetime.datetime(
            2026, 9, 23, tzinfo=datetime.UTC
        ) + datetime.timedelta(seconds=len(list(output.glob("batch_result.*.json"))))
        return await batch.run_batch(**kwargs)

    monkeypatch.setattr(cli, "run_batch", run_batch_with_distinct_start)
    arguments = [
        "5703tts",
        "--input",
        str(input_path),
        "--output",
        str(output),
        "--config",
        str(config_path),
        "--log-dir",
        str(tmp_path / "logs"),
        "--speaker-sidecar",
        str(sidecar_path),
    ]
    if policy is not None:
        arguments.extend(["--exclusion-policy", str(policy_path)])
    if resume:
        arguments.append("--resume")
    monkeypatch.setattr(sys, "argv", arguments)

    exit_code = asyncio.run(cli.main())
    manifest = json.loads((output / "batch_result.json").read_text(encoding="utf-8"))
    assert not (output / ".batch_result.json.tmp").exists()
    return exit_code, manifest, attempted


def _policy(*dialogue_ids: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "exclusions": [
            {
                "dialogue_id": dialogue_id,
                "reason": f"Synthetic issue {dialogue_id}",
                "provenance": {"ticket": dialogue_id.lower()},
            }
            for dialogue_id in dialogue_ids
        ],
    }


def _legacy_record(dialogue_id: str) -> dict[str, Any]:
    return {
        "dialogue_id": dialogue_id,
        "turns": [
            {
                "turn_id": 1,
                "speaker": "caller",
                "text": "Legacy",
                "label": "normal",
            }
        ],
    }


def test_jsonl_record_failures_continue_and_manifest_has_no_synthetic_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(
        input_path,
        ["", _record("A"), "{broken", "[1, 2]", '{"text":"missing id"}', _record("B")],
    )

    exit_code, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=_sidecar(("A", "B")),
    )

    assert exit_code == 1
    assert attempted == ["A", "B"]
    assert manifest["manifest_version"] == "2.0"
    assert manifest["summary"] == {
        "total_valid_records": 2,
        "rendered": 2,
        "resumed": 0,
        "excluded": 0,
        "failed": 3,
        "input_failures": 3,
    }
    failures = [item for item in manifest["results"] if item["action"] == "input_error"]
    assert len(failures) == 3
    assert all(item["dialogue_id"] is None for item in failures)
    assert [item["source"]["line_number"] for item in failures] == [3, 4, 5]
    assert [
        item["dialogue_id"]
        for item in manifest["results"]
        if item["status"] == "success"
    ] == ["A", "B"]


def test_explicit_final_json_record_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "one.json"
    input_path.write_text(json.dumps(_record("A")), encoding="utf-8")

    exit_code, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=_sidecar(("A",)),
    )

    assert exit_code == 0
    assert attempted == ["A"]
    assert manifest["results"][0]["source"]["format"] == "json"
    assert manifest["results"][0]["source"]["line_number"] is None


@pytest.mark.parametrize("across_containers", [False, True])
def test_duplicate_dialogue_ids_abort_before_any_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    across_containers: bool,
) -> None:
    if across_containers:
        input_path = tmp_path / "input"
        input_path.mkdir()
        (input_path / "a.json").write_text(json.dumps(_record("A")), encoding="utf-8")
        _write_jsonl(input_path / "b.jsonl", [_record("A")])
    else:
        input_path = tmp_path / "corpus.jsonl"
        _write_jsonl(input_path, [_record("A"), _record("A")])

    with pytest.raises(RuntimeError, match="Duplicate dialogue_id"):
        _invoke(
            tmp_path,
            monkeypatch,
            input_path=input_path,
            sidecar=_sidecar(("A",)),
        )

    assert not (tmp_path / "output").exists()


def test_mixed_batch_rejects_legacy_record_without_routing_to_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input"
    input_path.mkdir()
    (input_path / "final.json").write_text(json.dumps(_record("A")), encoding="utf-8")
    (input_path / "legacy.json").write_text(
        json.dumps(
            {
                "dialogue_id": "legacy",
                "turns": [
                    {
                        "turn_id": 1,
                        "speaker": "caller",
                        "text": "legacy",
                        "label": "normal",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=_sidecar(("A",)),
    )

    assert exit_code == 1
    assert attempted == ["A"]
    rejected = next(
        item for item in manifest["results"] if item["dialogue_id"] == "legacy"
    )
    assert rejected["action"] == "input_error"
    assert "legacy/v0.2 input is no longer supported" in rejected["error"]["message"]


def test_directory_final_jsonl_renders_while_legacy_json_is_input_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input"
    input_path.mkdir()
    (input_path / "legacy.json").write_text(
        json.dumps(_legacy_record("legacy")), encoding="utf-8"
    )
    _write_jsonl(input_path / "final.jsonl", [_record("A")])
    exit_code, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=_sidecar(("A",)),
    )

    assert attempted == ["A"]
    assert exit_code == 1
    assert [item["action"] for item in manifest["results"]] == [
        "rendered",
        "input_error",
    ]


def test_directory_with_final_jsonl_only_uses_final_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input"
    input_path.mkdir()
    _write_jsonl(input_path / "final.jsonl", [_record("A")])

    exit_code, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=_sidecar(("A",)),
    )

    assert exit_code == 0
    assert attempted == ["A"]
    assert "batch_schema_family" not in manifest


@pytest.mark.parametrize(
    "dialogue_id",
    ["../escape", "../../escape", "/tmp/evil", "foo/bar", r"foo\bar", ".", ".."],
)
def test_unsafe_final_dialogue_id_aborts_whole_batch_before_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dialogue_id: str,
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("safe"), _record(dialogue_id)])
    cleanup_calls = []
    monkeypatch.setattr(
        batch,
        "_cleanup_managed_dialogue_artifacts",
        lambda *args: cleanup_calls.append(args),
    )

    with pytest.raises(RuntimeError, match="final dialogue_id"):
        _invoke(
            tmp_path,
            monkeypatch,
            input_path=input_path,
            sidecar=_sidecar(("safe", dialogue_id)),
        )

    assert not (tmp_path / "output").exists()
    assert cleanup_calls == []


@pytest.mark.parametrize(
    "dialogue_id", ["corpus_v1_000001", "dialogue-001", "dialogue_001"]
)
def test_safe_final_dialogue_id_is_one_direct_output_child(
    tmp_path: Path, dialogue_id: str
) -> None:
    output_root = tmp_path / "output"

    candidate = batch._safe_dialogue_output_path(output_root, dialogue_id)

    assert candidate == output_root.resolve() / dialogue_id
    assert candidate.parent == output_root.resolve()


def test_resume_artifact_lookup_rejects_unsafe_dialogue_id(tmp_path: Path) -> None:
    output_root = tmp_path / "output"

    with pytest.raises(RuntimeError, match="final dialogue_id"):
        batch._safe_dialogue_output_path(output_root, "../escape")

    assert not output_root.exists()


@pytest.mark.parametrize("target_kind", ["in-root", "outside", "dangling"])
def test_dialogue_directory_symlink_is_rejected_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    if target_kind == "in-root":
        target = output_root / "B"
    elif target_kind == "outside":
        target = tmp_path / "outside"
    else:
        target = tmp_path / "missing"
    if target_kind != "dangling":
        target.mkdir()
        (target / "turn_001.wav").write_bytes(b"target turn")
        (target / "keep.txt").write_text("preserve", encoding="utf-8")
    (output_root / "A").symlink_to(target, target_is_directory=True)
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A")])
    cleanup_calls = []
    monkeypatch.setattr(
        batch,
        "_cleanup_managed_dialogue_artifacts",
        lambda *args: cleanup_calls.append(args),
    )

    with pytest.raises(RuntimeError, match="must not be a symlink"):
        _invoke(
            tmp_path,
            monkeypatch,
            input_path=input_path,
            sidecar=_sidecar(("A",)),
        )

    assert cleanup_calls == []
    assert (output_root / "A").is_symlink()
    if target_kind != "dangling":
        assert (target / "turn_001.wav").read_bytes() == b"target turn"
        assert (target / "keep.txt").read_text(encoding="utf-8") == "preserve"


def test_output_root_symlink_allows_real_dialogue_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    (tmp_path / "output").symlink_to(real_output, target_is_directory=True)
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A")])

    exit_code, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=_sidecar(("A",)),
    )

    assert exit_code == 0
    assert attempted == ["A"]
    assert manifest["results"][0]["action"] == "rendered"
    assert (tmp_path / "output").is_symlink()
    assert (real_output / "A" / "A_clean.wav").is_file()


@pytest.mark.parametrize(
    ("name", "managed"),
    [
        ("turn_001.wav", True),
        ("turn_999.wav", True),
        ("turn_1000.wav", True),
        ("turn_001.higgs_raw.wav", True),
        ("turn_001.higgs_raw.wav.part", True),
        ("turn_001.higgs_processed.part.wav", True),
        ("A_clean.wav", True),
        ("A_telephone.wav", True),
        ("A_metadata.json", True),
        ("turn_000.wav", False),
        ("turn_0001.wav", False),
        ("turn_1.wav", False),
        ("turn_01.wav", False),
        ("turn_001.mp3", False),
        ("turn_001.higgs_notes.wav", False),
        ("turn_backup.wav", False),
        ("turn_abc.wav", False),
        ("A_notes.json", False),
        ("other_clean.wav", False),
        ("keep.wav", False),
    ],
)
def test_managed_dialogue_artifact_matching(name: str, managed: bool) -> None:
    assert batch._is_managed_dialogue_artifact(name, "A") is managed


def test_cleanup_removes_only_managed_direct_child_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "output" / "A"
    out_dir.mkdir(parents=True)
    managed = {
        "turn_001.wav",
        "turn_005.wav",
        "turn_1000.wav",
        "turn_002.higgs_raw.wav",
        "turn_002.higgs_raw.wav.part",
        "turn_002.higgs_processed.part.wav",
        "A_clean.wav",
        "A_telephone.wav",
        "A_metadata.json",
    }
    for name in managed:
        (out_dir / name).write_bytes(b"old")
    keep = out_dir / "keep.txt"
    keep.write_text("preserve", encoding="utf-8")
    nested = out_dir / "nested"
    nested.mkdir()
    (nested / "turn_999.wav").write_bytes(b"nested")
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"outside")
    symlink = out_dir / "turn_007.wav"
    symlink.symlink_to(outside)

    batch._cleanup_managed_dialogue_artifacts(out_dir, "A")

    assert all(not (out_dir / name).exists() for name in managed)
    assert not symlink.exists()
    assert not symlink.is_symlink()
    assert outside.read_bytes() == b"outside"
    assert keep.read_text(encoding="utf-8") == "preserve"
    assert (nested / "turn_999.wav").read_bytes() == b"nested"


def test_cleanup_is_a_noop_for_missing_dialogue_directory(tmp_path: Path) -> None:
    batch._cleanup_managed_dialogue_artifacts(tmp_path / "missing", "A")


def test_missing_nonexcluded_sidecar_entry_aborts_before_any_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A"), _record("B")])

    with pytest.raises(RuntimeError, match="missing dialogue_id entries: B"):
        _invoke(
            tmp_path,
            monkeypatch,
            input_path=input_path,
            sidecar=_sidecar(("A",)),
        )

    assert not (tmp_path / "output").exists()


def test_exclusion_transitions_do_not_render_or_resume_excluded_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A"), _record("B")])

    first_exit, first, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=_sidecar(("A",)),
        policy=_policy("B"),
    )
    assert first_exit == 0
    assert attempted == ["A"]
    assert first["summary"]["excluded"] == 1
    excluded = first["results"][1]
    assert excluded["action"] == "excluded_known_issue"
    assert excluded["status"] == "excluded"
    assert excluded["exclusion"]["reason"] == "Synthetic issue B"
    assert excluded["exclusion"]["provenance"] == {"ticket": "b"}
    assert not (tmp_path / "output" / "B").exists()

    _, second, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=_sidecar(("A",)),
        policy=_policy("B"),
        resume=True,
    )
    assert attempted == []
    assert [item["action"] for item in second["results"]] == [
        "resumed",
        "excluded_known_issue",
    ]

    _, third, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=_sidecar(("A", "B")),
        policy=_policy(),
        resume=True,
    )
    assert attempted == ["B"]
    assert [item["action"] for item in third["results"]] == ["resumed", "rendered"]

    _, fourth, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=_sidecar(("A",)),
        policy=_policy("B"),
        resume=True,
    )
    assert attempted == []
    assert fourth["results"][1]["action"] == "excluded_known_issue"
    assert (tmp_path / "output" / "B" / "B_clean.wav").is_file()


def test_unrelated_assignment_and_policy_changes_do_not_invalidate_dialogue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A")])
    sidecar = _sidecar(("A",))
    _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        policy=_policy(),
    )
    changed_sidecar = deepcopy(sidecar)
    changed_sidecar["assignment_sha256"] = "f" * 64

    _, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=changed_sidecar,
        policy=_policy("UNRELATED"),
        resume=True,
    )

    assert attempted == []
    assert manifest["results"][0]["action"] == "resumed"


def test_record_edit_and_per_dialogue_reference_change_only_rerender_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    records = [_record("A"), _record("B"), _record("C")]
    _write_jsonl(input_path, records)
    sidecar = _sidecar(("A", "B", "C"))

    _, _, attempted = _invoke(
        tmp_path, monkeypatch, input_path=input_path, sidecar=sidecar
    )
    assert attempted == ["A", "B", "C"]

    _write_jsonl(input_path, ["", records[2], records[0], records[1]])
    _, moved, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        resume=True,
    )
    assert attempted == []
    assert all(item["action"] == "resumed" for item in moved["results"])
    assert (
        next(item for item in moved["results"] if item["dialogue_id"] == "A")["source"][
            "line_number"
        ]
        == 3
    )

    _write_jsonl(input_path, [records[0], _record("B", "B changed"), records[2]])
    _, edited, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        resume=True,
    )
    assert attempted == ["B"]
    assert [item["action"] for item in edited["results"]] == [
        "resumed",
        "rendered",
        "resumed",
    ]

    changed_sidecar = _sidecar(
        ("A", "B", "C"), b_speaker="spk_B_new", b_reference_sha="9" * 64
    )
    _, reference_changed, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=changed_sidecar,
        resume=True,
    )
    assert attempted == ["B"]
    assert [item["action"] for item in reference_changed["results"]] == [
        "resumed",
        "rendered",
        "resumed",
    ]


@pytest.mark.parametrize("global_field", ["registry", "active"])
def test_global_speaker_asset_provenance_change_does_not_rerender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, global_field: str
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A"), _record("B")])
    baseline = _sidecar(("A", "B"))
    _invoke(tmp_path, monkeypatch, input_path=input_path, sidecar=baseline)
    changed = _sidecar(
        ("A", "B"),
        registry_sha="d" * 64 if global_field == "registry" else "a" * 64,
        active_sha="e" * 64 if global_field == "active" else "b" * 64,
    )

    _, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=changed,
        resume=True,
    )

    assert attempted == []
    assert [item["action"] for item in manifest["results"]] == [
        "resumed",
        "resumed",
    ]


@pytest.mark.parametrize(
    "unsafe_previous",
    ["failed", "missing_fingerprint", "excluded"],
)
def test_unsafe_previous_final_entries_rerender(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_previous: str,
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A")])
    sidecar = _sidecar(("A",))
    _invoke(tmp_path, monkeypatch, input_path=input_path, sidecar=sidecar)
    manifest_path = tmp_path / "output" / "batch_result.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = previous["results"][0]
    if unsafe_previous == "failed":
        entry["status"] = "failed"
        entry["action"] = "render_failed"
    elif unsafe_previous == "missing_fingerprint":
        entry.pop("render_fingerprint")
    else:
        entry["status"] = "excluded"
        entry["action"] = "excluded_known_issue"
        entry["render_fingerprint"] = None
    manifest_path.write_text(json.dumps(previous), encoding="utf-8")

    _, current, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        resume=True,
    )

    assert attempted == ["A"]
    assert current["results"][0]["action"] == "rendered"


@pytest.mark.parametrize("missing", ["clean.wav", "telephone.wav", "metadata.json"])
def test_missing_final_artifact_rerenders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A")])
    sidecar = _sidecar(("A",))
    _invoke(tmp_path, monkeypatch, input_path=input_path, sidecar=sidecar)
    (tmp_path / "output" / "A" / f"A_{missing}").unlink()

    _, _, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        resume=True,
    )

    assert attempted == ["A"]


@pytest.mark.parametrize(
    "contents",
    [
        pytest.param("{broken", id="malformed"),
        pytest.param('{"dialogue_id":"A","dialogue_id":"A"}', id="duplicate-key"),
        pytest.param('{"invalid":NaN}', id="non-json-constant"),
    ],
)
def test_invalid_metadata_json_rerenders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: str,
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A")])
    sidecar = _sidecar(("A",))
    _invoke(tmp_path, monkeypatch, input_path=input_path, sidecar=sidecar)
    metadata_path = tmp_path / "output" / "A" / "A_metadata.json"
    metadata_path.write_text(contents, encoding="utf-8")

    _, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        resume=True,
    )

    assert attempted == ["A"]
    assert manifest["results"][0]["action"] == "rendered"


@pytest.mark.parametrize(
    ("keys", "value"),
    [
        pytest.param(("dialogue_id",), "B", id="wrong-dialogue"),
        pytest.param(("provenance", "record_sha256"), "wrong", id="wrong-source-sha"),
        pytest.param(("clean_audio",), "wrong.wav", id="wrong-clean-name"),
        pytest.param(("telephone_audio",), "wrong.wav", id="wrong-telephone-name"),
        pytest.param(("tts", "engine"), "cosyvoice", id="wrong-engine"),
        pytest.param(
            ("tts", "backend_identity", "backend"),
            "cosyvoice",
            id="wrong-backend",
        ),
        pytest.param(
            ("tts", "control_mapping", "implementation_id"),
            "wrong",
            id="wrong-metadata-implementation",
        ),
        pytest.param(
            (
                "tts",
                "backend_identity",
                "control_mapping",
                "implementation_id",
            ),
            "wrong",
            id="wrong-backend-implementation",
        ),
    ],
)
def test_metadata_identity_mismatch_rerenders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    keys: tuple[str | int, ...],
    value: Any,
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A")])
    sidecar = _sidecar(("A",))
    _invoke(tmp_path, monkeypatch, input_path=input_path, sidecar=sidecar)
    _change_metadata_value(tmp_path / "output" / "A" / "A_metadata.json", keys, value)

    _, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        resume=True,
    )

    assert attempted == ["A"]
    assert manifest["results"][0]["action"] == "rendered"


def test_missing_turn_wav_rerenders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A")])
    sidecar = _sidecar(("A",))
    _invoke(tmp_path, monkeypatch, input_path=input_path, sidecar=sidecar)
    (tmp_path / "output" / "A" / "turn_001.wav").unlink()

    _, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        resume=True,
    )

    assert attempted == ["A"]
    assert manifest["results"][0]["action"] == "rendered"


@pytest.mark.parametrize(
    "filename",
    ["turn_001.wav", "A_clean.wav", "A_telephone.wav"],
)
def test_corrupt_wav_rerenders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A")])
    sidecar = _sidecar(("A",))
    _invoke(tmp_path, monkeypatch, input_path=input_path, sidecar=sidecar)
    (tmp_path / "output" / "A" / filename).write_bytes(b"not a wav")

    _, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        resume=True,
    )

    assert attempted == ["A"]
    assert manifest["results"][0]["action"] == "rendered"


@pytest.mark.parametrize("turn_audio", ["wrong.wav", "../other.wav"])
def test_unexpected_metadata_turn_filename_rerenders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    turn_audio: str,
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A")])
    sidecar = _sidecar(("A",))
    _invoke(tmp_path, monkeypatch, input_path=input_path, sidecar=sidecar)
    _change_metadata_value(
        tmp_path / "output" / "A" / "A_metadata.json",
        ("turns", 0, "execution", "turn_audio"),
        turn_audio,
    )

    _, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        resume=True,
    )

    assert attempted == ["A"]
    assert manifest["results"][0]["action"] == "rendered"


def test_non_v2_manifest_is_rejected_by_the_only_resume_system(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A")])
    output = tmp_path / "output"
    output.mkdir()
    (output / "batch_result.json").write_text(
        json.dumps(
            {
                "status": "success",
                "results": [
                    {
                        "dialogue_id": "A",
                        "input_path": str(input_path),
                        "status": "success",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="malformed batch result"):
        _invoke(
            tmp_path,
            monkeypatch,
            input_path=input_path,
            sidecar=_sidecar(("A",)),
            resume=True,
        )


def test_final_v2_manifest_still_resumes_final_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "final.jsonl"
    sidecar = _sidecar(("A",))
    _write_jsonl(input_path, [_record("A")])
    _invoke(tmp_path, monkeypatch, input_path=input_path, sidecar=sidecar)
    out_dir = tmp_path / "output" / "A"
    managed_before = {
        path.name: path.read_bytes() for path in out_dir.iterdir() if path.is_file()
    }
    sentinel = out_dir / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    _, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        resume=True,
    )

    assert attempted == []
    assert manifest["results"][0]["action"] == "resumed"
    assert {
        path.name: path.read_bytes()
        for path in out_dir.iterdir()
        if path.is_file() and path != sentinel
    } == managed_before
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_corrupt_resume_candidate_cleans_stale_artifacts_before_rerender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "final.jsonl"
    sidecar = _sidecar(("A",))
    _write_jsonl(input_path, [_record("A")])
    _invoke(tmp_path, monkeypatch, input_path=input_path, sidecar=sidecar)
    out_dir = tmp_path / "output" / "A"
    (out_dir / "turn_999.wav").write_bytes(b"stale")
    (out_dir / "A_clean.wav").write_bytes(b"corrupt")
    sentinel = out_dir / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    _, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        resume=True,
    )

    assert attempted == ["A"]
    assert manifest["results"][0]["action"] == "rendered"
    assert not (out_dir / "turn_999.wav").exists()
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_shorter_rerender_removes_surplus_turn_wavs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "final.jsonl"
    sidecar = _sidecar(("A",))
    _write_jsonl(input_path, [_record_with_turn_count("A", 6)])
    _invoke(tmp_path, monkeypatch, input_path=input_path, sidecar=sidecar)
    out_dir = tmp_path / "output" / "A"
    sentinel = out_dir / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    _write_jsonl(input_path, [_record_with_turn_count("A", 4)])

    _, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        resume=True,
    )

    assert attempted == ["A"]
    assert manifest["results"][0]["action"] == "rendered"
    assert sorted(path.name for path in out_dir.glob("turn_*.wav")) == [
        "turn_001.wav",
        "turn_002.wav",
        "turn_003.wav",
        "turn_004.wav",
    ]
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_cleanup_failure_fails_dialogue_without_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "final.jsonl"
    sidecar = _sidecar(("A",))
    _write_jsonl(input_path, [_record("A")])
    _invoke(tmp_path, monkeypatch, input_path=input_path, sidecar=sidecar)
    out_dir = tmp_path / "output" / "A"
    (out_dir / "turn_999.wav").mkdir()
    _write_jsonl(input_path, [_record("A", "changed")])

    exit_code, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        resume=True,
    )

    assert exit_code == 1
    assert attempted == []
    entry = manifest["results"][0]
    assert entry["action"] == "render_failed"
    assert entry["status"] == "failed"
    assert entry["error"]["type"] == "RuntimeError"
    assert "Managed artifact path is not a file" in entry["error"]["message"]
    assert (out_dir / "turn_999.wav").is_dir()


def test_failed_rerender_does_not_leave_old_final_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "final.jsonl"
    sidecar = _sidecar(("A",))
    _write_jsonl(input_path, [_record("A")])
    _invoke(tmp_path, monkeypatch, input_path=input_path, sidecar=sidecar)
    out_dir = tmp_path / "output" / "A"
    sentinel = out_dir / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    _write_jsonl(input_path, [_record("A", "changed")])

    exit_code, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        resume=True,
        failures={"A"},
    )

    assert exit_code == 1
    assert attempted == ["A"]
    assert manifest["results"][0]["action"] == "render_failed"
    assert not any(
        batch._is_managed_dialogue_artifact(path.name, "A")
        for path in out_dir.iterdir()
    )
    assert sentinel.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("engine", ["higgs", "cosyvoice"])
def test_live_selected_reference_mutation_prevents_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    input_path = tmp_path / "final.jsonl"
    sidecar = _sidecar(("A",))
    _write_jsonl(input_path, [_record("A")])
    _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        engine=engine,
    )

    _, unchanged, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        engine=engine,
        resume=True,
    )
    assert attempted == []
    assert unchanged["results"][0]["action"] == "resumed"

    field = "reference_wav" if engine == "higgs" else "prompt_wav"
    declared = sidecar["dialogues"][0]["roles"]["caller"][f"{engine}_reference"][field]
    (tmp_path / declared).write_bytes(b"mutated live reference bytes")

    _, mutated, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        engine=engine,
        resume=True,
    )
    assert attempted == ["A"]
    assert mutated["results"][0]["action"] == "rendered"


@pytest.mark.parametrize(
    ("before", "after"),
    [("higgs", "cosyvoice"), ("cosyvoice", "higgs")],
)
def test_engine_switch_invalidates_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    before: str,
    after: str,
) -> None:
    input_path = tmp_path / "final.jsonl"
    sidecar = _sidecar(("A",))
    _write_jsonl(input_path, [_record("A")])
    _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        engine=before,
    )

    _, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        engine=after,
        resume=True,
    )

    assert attempted == ["A"]
    assert manifest["results"][0]["action"] == "rendered"


def test_cosyvoice_prompt_text_change_invalidates_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "final.jsonl"
    sidecar = _sidecar(("A",))
    _write_jsonl(input_path, [_record("A")])
    _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        engine="cosyvoice",
    )
    changed = deepcopy(sidecar)
    changed["dialogues"][0]["roles"]["caller"]["cosyvoice_reference"]["prompt_text"] = (
        "Changed prompt text"
    )

    _, _, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=changed,
        engine="cosyvoice",
        resume=True,
    )

    assert attempted == ["A"]


@pytest.mark.parametrize("engine", ["higgs", "cosyvoice"])
def test_selected_reference_path_and_sha_change_invalidate_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    input_path = tmp_path / "final.jsonl"
    sidecar = _sidecar(("A",))
    _write_jsonl(input_path, [_record("A")])
    _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        engine=engine,
    )
    changed = deepcopy(sidecar)
    reference = changed["dialogues"][0]["roles"]["caller"][f"{engine}_reference"]
    path_field = "reference_wav" if engine == "higgs" else "prompt_wav"
    reference[path_field] = f"refs/A-caller-{engine}-changed.wav"
    reference["sha256"] = _reference_sha(reference[path_field])

    _, _, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=changed,
        engine=engine,
        resume=True,
    )

    assert attempted == ["A"]


@pytest.mark.parametrize("engine", ["higgs", "cosyvoice"])
def test_unselected_reference_change_does_not_invalidate_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    input_path = tmp_path / "final.jsonl"
    sidecar = _sidecar(("A",))
    _write_jsonl(input_path, [_record("A")])
    _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        engine=engine,
    )
    changed = deepcopy(sidecar)
    unselected = "cosyvoice" if engine == "higgs" else "higgs"
    changed["dialogues"][0]["roles"]["caller"][f"{unselected}_reference"] = {
        "malformed": True
    }

    _, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=changed,
        engine=engine,
        resume=True,
    )

    assert attempted == []
    assert manifest["results"][0]["action"] == "resumed"


@pytest.mark.parametrize("engine", ["higgs", "cosyvoice"])
def test_malformed_selected_reference_is_not_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    input_path = tmp_path / "final.jsonl"
    sidecar = _sidecar(("A",))
    _write_jsonl(input_path, [_record("A")])
    _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        engine=engine,
    )
    changed = deepcopy(sidecar)
    changed["dialogues"][0]["roles"]["caller"][f"{engine}_reference"] = {
        "malformed": True
    }

    _, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=changed,
        engine=engine,
        resume=True,
    )

    assert attempted == ["A"]
    assert manifest["results"][0]["action"] == "rendered"
    assert manifest["results"][0]["render_fingerprint"] is None


@pytest.mark.parametrize("engine", ["higgs", "cosyvoice"])
def test_unselected_backend_config_change_does_not_invalidate_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    input_path = tmp_path / "final.jsonl"
    sidecar = _sidecar(("A",))
    config = _config(engine)
    _write_jsonl(input_path, [_record("A")])
    _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        config=config,
    )
    changed = deepcopy(config)
    unselected = "cosyvoice" if engine == "higgs" else "higgs"
    changed["tts"][unselected]["model_dir"] = "models/unselected-change"

    _, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        config=changed,
        resume=True,
    )

    assert attempted == []
    assert manifest["results"][0]["action"] == "resumed"


@pytest.mark.parametrize("engine", ["higgs", "cosyvoice"])
def test_invalid_unselected_backend_config_does_not_block_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    input_path = tmp_path / "final.jsonl"
    config = _config(engine)
    unselected = "cosyvoice" if engine == "higgs" else "higgs"
    config["tts"][unselected] = "intentionally invalid and unselected"
    _write_jsonl(input_path, [_record("A")])

    exit_code, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=_sidecar(("A",)),
        config=config,
    )

    assert exit_code == 0
    assert attempted == ["A"]
    assert manifest["results"][0]["action"] == "rendered"


@pytest.mark.parametrize("change", ["shared", "selected"])
def test_render_affecting_config_change_invalidates_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    input_path = tmp_path / "final.jsonl"
    sidecar = _sidecar(("A",))
    config = _config("higgs")
    _write_jsonl(input_path, [_record("A")])
    _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        config=config,
    )
    changed = deepcopy(config)
    if change == "shared":
        changed["fade_ms"] = 9
    else:
        changed["tts"]["higgs"]["model_dir"] = "models/selected-change"

    _, _, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        config=changed,
        resume=True,
    )

    assert attempted == ["A"]


def test_final_sidecar_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "sidecar.json"
    sidecar_path.write_text('{"dialogues":[],"dialogues":[]}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="duplicate object key"):
        batch._load_sidecar(sidecar_path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_final_sidecar_rejects_non_json_constants(
    tmp_path: Path, constant: str
) -> None:
    sidecar_path = tmp_path / "sidecar.json"
    sidecar_path.write_text(
        f'{{"dialogues":[],"invalid":{constant}}}', encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="invalid JSON constant"):
        batch._load_sidecar(sidecar_path)


def test_backend_semantic_identity_change_rerenders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A")])
    sidecar = _sidecar(("A",))
    _invoke(tmp_path, monkeypatch, input_path=input_path, sidecar=sidecar)
    original = batch_identity._backend_static_identity

    def changed(config: dict) -> dict:
        identity = deepcopy(original(config))
        identity["control_mapping"]["implementation_id"] = "changed-for-test"
        return identity

    monkeypatch.setattr(batch_identity, "_backend_static_identity", changed)
    _, _, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        resume=True,
    )

    assert attempted == ["A"]


def test_final_manifest_records_semantic_and_provenance_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, ["", _record("A")])
    sidecar = _sidecar(("A",))

    exit_code, manifest, _ = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
    )

    assert exit_code == 0
    assert manifest["manifest_version"] == "2.0"
    assert "batch_schema_family" not in manifest
    assert manifest["backend_identity_sha256"]
    controlled = manifest["batch_provenance"]["controlled_tts"]
    assert controlled["mapping_version"] == "controlled_tts_v1"
    assert controlled["implementation_id"] == "controlled_tts_v1_prod_1"
    assert controlled["contract_sha256"] == (
        "4e787a5363d01e5454a8031c59bb1d4bb454ee37d44ccf262d471c7ce30ba799"
    )
    assert manifest["batch_provenance"]["exclusion_policy"] == {
        "schema_version": None,
        "policy_sha256": None,
    }
    entry = manifest["results"][0]
    assert entry["source"]["line_number"] == 2
    assert entry["source"]["record_sha256"]
    assert entry["render_fingerprint"]
    assert entry["action"] == "rendered"
    assert entry["status"] == "success"
    assert entry["error"] is None
    assert entry["exclusion"] is None


def test_batch_attempt_manifests_preserve_first_result_and_update_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A")])
    sidecar = _sidecar(("A",))
    output = tmp_path / "output"

    first_exit, first, _ = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        failures={"A"},
    )
    assert first_exit == 1
    first_attempts = list(output.glob("batch_result.*.json"))
    assert len(first_attempts) == 1
    first_attempt = first_attempts[0]
    first_bytes = first_attempt.read_bytes()
    assert "." not in first["started_at"]
    assert first_attempt.name == (
        "batch_result."
        + first["started_at"].replace(":", "-").replace("+", "-")
        + ".json"
    )
    assert json.loads(first_bytes) == first
    assert json.loads((output / "batch_result.json").read_bytes()) == first

    second_exit, second, _ = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        resume=True,
    )
    assert second_exit == 0
    attempts = list(output.glob("batch_result.*.json"))
    assert len(attempts) == 2
    assert first_attempt.read_bytes() == first_bytes
    second_attempt = next(path for path in attempts if path != first_attempt)
    assert json.loads(second_attempt.read_bytes()) == second
    assert json.loads((output / "batch_result.json").read_bytes()) == second
    assert first["summary"]["failed"] == 1
    assert second["summary"]["failed"] == 0


def test_batch_attempt_collision_preserves_evidence_and_latest(tmp_path: Path) -> None:
    first = {"started_at": "2026-09-23T07:48:12+00:00", "status": "failure"}
    latest = batch._write_batch_result(first, tmp_path)
    (attempt,) = tmp_path.glob("batch_result.*.json")
    first_bytes = attempt.read_bytes()
    assert attempt.name == "batch_result.2026-09-23T07-48-12-00-00.json"
    assert latest.read_bytes() == first_bytes

    with pytest.raises(RuntimeError, match="attempt manifest already exists"):
        batch._write_batch_result({**first, "status": "success"}, tmp_path)

    assert attempt.read_bytes() == first_bytes
    assert latest.read_bytes() == first_bytes
    assert not list(tmp_path.glob(".batch_result.*.tmp"))


def test_latest_manifest_does_not_share_attempt_file_bytes(tmp_path: Path) -> None:
    result = {"started_at": "2026-09-23T07:48:12+00:00", "status": "failure"}
    latest = batch._write_batch_result(result, tmp_path)
    (attempt,) = tmp_path.glob("batch_result.*.json")
    first_bytes = attempt.read_bytes()

    latest.write_text("changed latest", encoding="utf-8")

    assert attempt.read_bytes() == first_bytes


def test_render_failure_records_expected_output_even_when_result_out_dir_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A")])

    exit_code, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=_sidecar(("A",)),
        failures={"A"},
    )

    assert exit_code == 1
    assert attempted == ["A"]
    entry = manifest["results"][0]
    assert entry["action"] == "render_failed"
    assert entry["output_dir"] == str(tmp_path / "output" / "A")
    assert (tmp_path / "output" / "A").is_dir()


def test_final_batch_cosyvoice_engine_is_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A")])

    exit_code, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=_sidecar(("A",)),
        engine="cosyvoice",
    )

    assert exit_code == 0
    assert attempted == ["A"]
    assert manifest["backend"] == "cosyvoice"


def test_final_batch_unsupported_engine_fails_before_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A")])

    with pytest.raises(RuntimeError, match="supports only tts.engine='higgs' or"):
        _invoke(
            tmp_path,
            monkeypatch,
            input_path=input_path,
            sidecar=_sidecar(("A",)),
            engine="unsupported",
        )

    assert not (tmp_path / "output").exists()


def test_legacy_json_is_rejected_without_calling_either_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "legacy.json").write_text(
        json.dumps(_legacy_record("legacy-A")), encoding="utf-8"
    )
    exit_code, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_root,
        sidecar=_sidecar(("legacy-A",)),
    )

    assert exit_code == 1
    assert attempted == []
    assert manifest["results"][0]["action"] == "input_error"
    assert (
        "legacy/v0.2 input is no longer supported"
        in manifest["results"][0]["error"]["message"]
    )


def test_v02_json_is_rejected_before_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "v02.json"
    raw = _legacy_record("v02-A")
    raw["schema_version"] = "0.2"
    raw["turns"][0]["acoustic_spec"] = {
        "rate": "normal",
        "pause_before_ms": 0,
        "pause_after_ms": 0,
        "arousal": "medium",
        "coarse_affect": "neutral",
        "hesitations": 0,
        "long_pauses": 0,
    }
    input_path.write_text(json.dumps(raw), encoding="utf-8")

    exit_code, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=_sidecar(("v02-A",)),
    )

    assert exit_code == 1
    assert attempted == []
    assert manifest["results"][0]["action"] == "input_error"
    assert (
        "legacy/v0.2 input is no longer supported"
        in manifest["results"][0]["error"]["message"]
    )


def test_production_cli_has_no_legacy_dispatch_call_path() -> None:
    source = inspect.getsource(cli.main) + inspect.getsource(batch.run_batch)

    for forbidden in (
        "_is_final_batch",
        "load_and_validate",
        "legacy_run_dialogue",
        "_discover_legacy_json_files",
    ):
        assert forbidden not in source
