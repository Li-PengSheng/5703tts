from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tts5703 import batch_identity, cli
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
                "voice_map": {},
            },
            "cosyvoice": {
                "python_bin": "python",
                "repo_dir": "third_party/CosyVoice",
                "model_dir": "models/CosyVoice",
                "voice_map": {},
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


def _write_success_artifacts(output_root: Path, dialogue_id: str) -> Path:
    out_dir = output_root / dialogue_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("clean.wav", "telephone.wav", "metadata.json"):
        (out_dir / f"{dialogue_id}_{suffix}").write_bytes(b"complete")
    return out_dir


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

    async def fake_run_final(
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
        out_dir = _write_success_artifacts(output_root, dialogue_id)
        return PipelineResult(dialogue_id, "success", out_dir=out_dir)

    monkeypatch.setattr(cli, "configure_logging", lambda *_: None)
    monkeypatch.setattr(cli, "load_config", lambda _: deepcopy(loaded_config))
    monkeypatch.setattr(cli, "run_final_dialogue", fake_run_final)
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


def _invoke_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    input_path: Path,
    resume: bool = False,
) -> tuple[int, dict[str, Any], list[str]]:
    output = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("legacy config\n", encoding="utf-8")
    attempted: list[str] = []

    def fake_validate(path: Path, _: dict) -> SimpleNamespace:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return SimpleNamespace(dialogue_id=raw["dialogue_id"])

    async def fake_run(path: Path, config: dict, output_root: Path) -> PipelineResult:
        dialogue_id = json.loads(path.read_text(encoding="utf-8"))["dialogue_id"]
        attempted.append(path.name)
        return PipelineResult(dialogue_id, "success", out_dir=output_root / dialogue_id)

    async def unexpected_final(*args: object, **kwargs: object) -> PipelineResult:
        raise AssertionError("legacy batch called run_final_dialogue")

    monkeypatch.setattr(cli, "configure_logging", lambda *_: None)
    monkeypatch.setattr(cli, "load_config", lambda _: {"tts": {"engine": "kokoro"}})
    monkeypatch.setattr(cli, "load_and_validate", fake_validate)
    monkeypatch.setattr(cli, "run_dialogue", fake_run)
    monkeypatch.setattr(cli, "run_final_dialogue", unexpected_final)
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
    ]
    if resume:
        arguments.append("--resume")
    monkeypatch.setattr(sys, "argv", arguments)
    exit_code = asyncio.run(cli.main())
    manifest = json.loads((output / "batch_result.json").read_text(encoding="utf-8"))
    return exit_code, manifest, attempted


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


def test_mixed_final_and_legacy_batch_aborts_before_render(
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

    with pytest.raises(RuntimeError, match="Cannot mix final nested"):
        _invoke(
            tmp_path,
            monkeypatch,
            input_path=input_path,
            sidecar=_sidecar(("A",)),
        )

    assert not (tmp_path / "output").exists()


def test_legacy_directory_ignores_stray_nonfinal_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input"
    input_path.mkdir()
    (input_path / "legacy_a.json").write_text(
        json.dumps(_legacy_record("legacy-A")), encoding="utf-8"
    )
    (input_path / "legacy_b.json").write_text(
        json.dumps(_legacy_record("legacy-B")), encoding="utf-8"
    )
    _write_jsonl(input_path / "stray.jsonl", [_legacy_record("stray")])

    exit_code, _, attempted = _invoke_legacy(
        tmp_path, monkeypatch, input_path=input_path
    )

    assert exit_code == 0
    assert attempted == ["legacy_a.json", "legacy_b.json"]


def test_directory_final_jsonl_and_legacy_json_rejected_before_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input"
    input_path.mkdir()
    (input_path / "legacy.json").write_text(
        json.dumps(_legacy_record("legacy")), encoding="utf-8"
    )
    _write_jsonl(input_path / "final.jsonl", [_record("A")])
    legacy_calls: list[Path] = []

    async def unexpected_legacy(path: Path, *_: object) -> PipelineResult:
        legacy_calls.append(path)
        raise AssertionError("mixed batch called run_dialogue")

    monkeypatch.setattr(cli, "run_dialogue", unexpected_legacy)
    with pytest.raises(RuntimeError, match="Cannot mix final nested"):
        _invoke(
            tmp_path,
            monkeypatch,
            input_path=input_path,
            sidecar=_sidecar(("A",)),
        )

    assert legacy_calls == []
    assert not (tmp_path / "output").exists()


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
    assert manifest["batch_schema_family"] == "final_nested"


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

    with pytest.raises(RuntimeError, match="final dialogue_id"):
        _invoke(
            tmp_path,
            monkeypatch,
            input_path=input_path,
            sidecar=_sidecar(("safe", dialogue_id)),
        )

    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "dialogue_id", ["corpus_v1_000001", "dialogue-001", "dialogue_001"]
)
def test_safe_final_dialogue_id_is_one_direct_output_child(
    tmp_path: Path, dialogue_id: str
) -> None:
    output_root = tmp_path / "output"

    candidate = cli._safe_dialogue_output_path(output_root, dialogue_id)

    assert candidate == output_root.resolve() / dialogue_id
    assert candidate.parent == output_root.resolve()


def test_resume_artifact_lookup_rejects_unsafe_dialogue_id(tmp_path: Path) -> None:
    output_root = tmp_path / "output"

    with pytest.raises(RuntimeError, match="final dialogue_id"):
        cli._final_artifacts_exist(output_root, "../escape")

    assert not output_root.exists()


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


def test_old_manifest_without_final_fingerprint_rerenders_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A")])
    output = tmp_path / "output"
    _write_success_artifacts(output, "A")
    output.mkdir(exist_ok=True)
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

    _, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=_sidecar(("A",)),
        resume=True,
    )

    assert attempted == ["A"]
    assert manifest["results"][0]["action"] == "rendered"


def test_legacy_resume_ignores_final_v2_manifest_and_renders_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_input = tmp_path / "final.jsonl"
    _write_jsonl(final_input, [_record("A")])
    _invoke(
        tmp_path,
        monkeypatch,
        input_path=final_input,
        sidecar=_sidecar(("A",)),
    )
    previous = json.loads(
        (tmp_path / "output" / "batch_result.json").read_text(encoding="utf-8")
    )
    assert previous["manifest_version"] == "2.0"

    legacy_input = tmp_path / "legacy"
    legacy_input.mkdir()
    (legacy_input / "legacy.json").write_text(
        json.dumps(_legacy_record("legacy-A")), encoding="utf-8"
    )
    exit_code, manifest, attempted = _invoke_legacy(
        tmp_path,
        monkeypatch,
        input_path=legacy_input,
        resume=True,
    )

    assert exit_code == 0
    assert attempted == ["legacy.json"]
    assert manifest["results"][0]["action"] == "rendered"


def test_final_v2_manifest_still_resumes_final_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "final.jsonl"
    sidecar = _sidecar(("A",))
    _write_jsonl(input_path, [_record("A")])
    _invoke(tmp_path, monkeypatch, input_path=input_path, sidecar=sidecar)

    _, manifest, attempted = _invoke(
        tmp_path,
        monkeypatch,
        input_path=input_path,
        sidecar=sidecar,
        resume=True,
    )

    assert attempted == []
    assert manifest["results"][0]["action"] == "resumed"


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
        cli._load_final_sidecar(sidecar_path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_final_sidecar_rejects_non_json_constants(
    tmp_path: Path, constant: str
) -> None:
    sidecar_path = tmp_path / "sidecar.json"
    sidecar_path.write_text(
        f'{{"dialogues":[],"invalid":{constant}}}', encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="invalid JSON constant"):
        cli._load_final_sidecar(sidecar_path)


def test_backend_semantic_identity_change_rerenders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_jsonl(input_path, [_record("A")])
    sidecar = _sidecar(("A",))
    _invoke(tmp_path, monkeypatch, input_path=input_path, sidecar=sidecar)
    original = batch_identity.final_controlled_tts_backend_identity

    def changed(config: dict, dialogue: object) -> dict:
        identity = deepcopy(original(config, dialogue))
        identity["control_mapping"]["implementation_id"] = "changed-for-test"
        return identity

    monkeypatch.setattr(
        batch_identity, "final_controlled_tts_backend_identity", changed
    )
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
    assert manifest["batch_schema_family"] == "final_nested"
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

    with pytest.raises(RuntimeError, match="require tts.engine='higgs' or"):
        _invoke(
            tmp_path,
            monkeypatch,
            input_path=input_path,
            sidecar=_sidecar(("A",)),
            engine="kokoro",
        )

    assert not (tmp_path / "output").exists()


def test_legacy_json_still_uses_run_dialogue_without_final_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    input_root.mkdir()
    config_path.write_text("legacy config\n", encoding="utf-8")
    legacy = {
        "dialogue_id": "legacy-A",
        "turns": [
            {
                "turn_id": 1,
                "speaker": "caller",
                "text": "Legacy",
                "label": "normal",
            }
        ],
    }
    (input_root / "legacy.json").write_text(json.dumps(legacy), encoding="utf-8")
    attempted: list[str] = []

    async def fake_run(path: Path, config: dict, output: Path) -> PipelineResult:
        attempted.append(path.name)
        return PipelineResult("legacy-A", "success", out_dir=output / "legacy-A")

    class Dialogue:
        dialogue_id = "legacy-A"

    monkeypatch.setattr(cli, "configure_logging", lambda *_: None)
    monkeypatch.setattr(cli, "load_config", lambda _: {"tts": {"engine": "kokoro"}})
    monkeypatch.setattr(cli, "load_and_validate", lambda *_: Dialogue())
    monkeypatch.setattr(cli, "run_dialogue", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "5703tts",
            "--input",
            str(input_root),
            "--output",
            str(output_root),
            "--config",
            str(config_path),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert asyncio.run(cli.main()) == 0
    assert attempted == ["legacy.json"]
