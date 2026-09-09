"""Offline batch-result and process-status tests for the production CLI."""

import asyncio
import hashlib
import json
import sys
import wave
from pathlib import Path

import pytest

from tts5703 import cli
from tts5703.config import ConfigError
from tts5703.pipeline import PipelineResult
from tts5703.validate import NormalizedDialogue, NormalizedTurn, ValidationError

_INPUT_BODY = "{}\n"
_CONFIG_TEXT = "tts:\n  engine: edge_tts\n"
_CONFIG = {"tts": {"engine": "edge_tts"}, "speaker_voice_map": {}}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _one_turn_dialogue(dialogue_id: str) -> NormalizedDialogue:
    return NormalizedDialogue(
        dialogue_id=dialogue_id,
        turns=[
            NormalizedTurn(
                turn_id=1,
                speaker="caller",
                text="hello",
                label="normal",
                rate="normal",
                pause_before_ms=0,
                pause_after_ms=0,
            )
        ],
    )


def _write_silent_wav(path: Path, duration_ms: int = 200) -> None:
    sample_rate = 8_000
    frame_count = sample_rate * duration_ms // 1_000
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\x00\x00" * frame_count)


def _complete_dialogue_output(output_root: Path, dialogue_id: str) -> Path:
    out_dir = output_root / dialogue_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_silent_wav(out_dir / f"{dialogue_id}_clean.wav")
    _write_silent_wav(out_dir / f"{dialogue_id}_telephone.wav")
    _write_silent_wav(out_dir / "turn_001.wav")
    metadata = {
        "dialogue_id": dialogue_id,
        "clean_audio": f"{dialogue_id}_clean.wav",
        "telephone_audio": f"{dialogue_id}_telephone.wav",
        "turns": [
            {
                "turn_id": 1,
                "speaker": "caller",
                "text": "hello",
                "label": "normal",
                "turn_audio": "turn_001.wav",
                "start_time": 0.0,
                "end_time": 0.2,
            }
        ],
    }
    (out_dir / f"{dialogue_id}_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return out_dir


def _run_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcomes: dict[str, PipelineResult],
    *,
    extra_args: list[str] | None = None,
    input_bodies: dict[str, str] | None = None,
    write_inputs: bool = True,
    config_text: str = _CONFIG_TEXT,
    complete_outputs: bool = False,
    previous_manifest: dict | None = None,
    attempted: list[str] | None = None,
    validate=None,
) -> tuple[int, dict, list[str]]:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    input_root.mkdir(exist_ok=True)
    output_root.mkdir(exist_ok=True)
    config_path.write_text(config_text, encoding="utf-8")
    if write_inputs:
        for filename in outcomes:
            body = (input_bodies or {}).get(filename, _INPUT_BODY)
            (input_root / filename).write_text(body, encoding="utf-8")
    if previous_manifest is not None:
        (output_root / "batch_result.json").write_text(
            json.dumps(previous_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if attempted is None:
        attempted = []

    async def fake_run_dialogue(
        path: Path, config: dict, output: Path
    ) -> PipelineResult:
        attempted.append(path.name)
        result = outcomes[path.name]
        if result.status == "success" and result.out_dir:
            result.out_dir.mkdir(parents=True, exist_ok=True)
            (result.out_dir / "preserved.txt").write_text(
                "complete\n", encoding="utf-8"
            )
            if complete_outputs and result.dialogue_id:
                _complete_dialogue_output(output, result.dialogue_id)
        return result

    def fake_load_and_validate(path: Path, config: dict) -> NormalizedDialogue:
        if validate is not None:
            return validate(path, config)
        result = outcomes[path.name]
        return _one_turn_dialogue(result.dialogue_id or path.stem)

    monkeypatch.setattr(cli, "load_config", lambda _: _CONFIG)
    monkeypatch.setattr(cli, "run_dialogue", fake_run_dialogue)
    monkeypatch.setattr(cli, "load_and_validate", fake_load_and_validate)
    monkeypatch.setattr(cli, "configure_logging", lambda *_: None)
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
            *(extra_args or []),
        ],
    )

    exit_code = asyncio.run(cli.main())
    manifest = json.loads(
        (output_root / "batch_result.json").read_text(encoding="utf-8")
    )
    assert not (output_root / ".batch_result.json.tmp").exists()
    return exit_code, manifest, attempted


def _success(tmp_path: Path, dialogue_id: str) -> PipelineResult:
    return PipelineResult(
        dialogue_id, "success", out_dir=tmp_path / "output" / dialogue_id
    )


def _failed(dialogue_id: str | None, message: str, error_type: str) -> PipelineResult:
    return PipelineResult(dialogue_id, "failed", message, error_type=error_type)


def _previous_entry(
    tmp_path: Path,
    filename: str,
    dialogue_id: str | None,
    status: str,
    *,
    input_sha256: str | None = _sha256_text(_INPUT_BODY),
    include_fingerprints: bool = True,
) -> dict:
    entry = {
        "dialogue_id": dialogue_id,
        "input_path": str(tmp_path / "input" / filename),
        "status": status,
        "output_dir": (
            str(tmp_path / "output" / dialogue_id) if status == "success" else None
        ),
    }
    if include_fingerprints:
        entry["input_sha256"] = input_sha256
        entry["action"] = "rendered"
    return entry


def _previous_manifest(
    tmp_path: Path,
    results: list[dict],
    *,
    status: str,
    include_fingerprints: bool = True,
    config_sha256: str | None = None,
    backend_identity: dict | None = None,
) -> dict:
    succeeded = sum(item["status"] == "success" for item in results)
    manifest = {
        "status": status,
        "backend": "edge_tts",
        "config_path": str(tmp_path / "config.yaml"),
        "input_root": str(tmp_path / "input"),
        "output_root": str(tmp_path / "output"),
        "dialogues_total": len(results),
        "dialogues_succeeded": succeeded,
        "dialogues_failed": len(results) - succeeded,
        "results": results,
    }
    if include_fingerprints:
        manifest["config_sha256"] = config_sha256 or _sha256_text(_CONFIG_TEXT)
        manifest["backend_identity"] = backend_identity or {"backend": "edge_tts"}
        manifest["resume_requested"] = False
    return manifest


def test_all_success_writes_manifest_and_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    outcomes = {
        "a.json": PipelineResult("A", "success", out_dir=output / "A"),
        "b.json": PipelineResult("B", "success", out_dir=output / "B"),
    }

    exit_code, manifest, attempted = _run_batch(tmp_path, monkeypatch, outcomes)

    assert attempted == ["a.json", "b.json"]
    assert exit_code == 0
    assert manifest["status"] == "success"
    assert manifest["dialogues_total"] == 2
    assert manifest["dialogues_succeeded"] == 2
    assert manifest["dialogues_failed"] == 0
    assert manifest["dialogues_rendered"] == 2
    assert manifest["dialogues_retried"] == 0
    assert manifest["dialogues_skipped"] == 0
    assert manifest["resume_requested"] is False
    assert manifest["config_sha256"] == _sha256_text(_CONFIG_TEXT)
    assert manifest["backend_identity"] == {"backend": "edge_tts"}
    assert [result["status"] for result in manifest["results"]] == [
        "success",
        "success",
    ]
    assert [result["action"] for result in manifest["results"]] == [
        "rendered",
        "rendered",
    ]
    assert [result["output_dir"] for result in manifest["results"]] == [
        str(output / "A"),
        str(output / "B"),
    ]
    assert all("error" not in result for result in manifest["results"])
    assert all(
        result["input_sha256"] == _sha256_text(_INPUT_BODY)
        for result in manifest["results"]
    )


def test_partial_failure_continues_and_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    outcomes = {
        "a.json": PipelineResult("A", "success", out_dir=output / "A"),
        "b.json": PipelineResult(
            "B",
            "failed",
            "Unexpected error: worker vanished",
            error_type="RuntimeError",
        ),
        "c.json": PipelineResult("C", "success", out_dir=output / "C"),
    }

    exit_code, manifest, attempted = _run_batch(tmp_path, monkeypatch, outcomes)

    assert attempted == ["a.json", "b.json", "c.json"]
    assert exit_code == 1
    assert manifest["status"] == "partial_failure"
    assert manifest["dialogues_succeeded"] == 2
    assert manifest["dialogues_failed"] == 1
    assert manifest["results"][1] == {
        "action": "rendered",
        "dialogue_id": "B",
        "error": {
            "message": "Unexpected error: worker vanished",
            "type": "RuntimeError",
        },
        "input_path": str(tmp_path / "input" / "b.json"),
        "input_sha256": _sha256_text(_INPUT_BODY),
        "output_dir": None,
        "status": "failed",
    }
    assert (output / "A" / "preserved.txt").exists()
    assert (output / "C" / "preserved.txt").exists()


def test_all_failed_records_details_and_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes = {
        "a.json": PipelineResult(
            None,
            "failed",
            "Input validation failed: malformed input",
            error_type="ValidationError",
        ),
        "b.json": PipelineResult(
            "B",
            "failed",
            "Backend control preflight failed: unsupported affect",
            error_type="BackendControlError",
        ),
    }

    exit_code, manifest, attempted = _run_batch(tmp_path, monkeypatch, outcomes)

    assert attempted == ["a.json", "b.json"]
    assert exit_code == 1
    assert manifest["status"] == "failure"
    assert manifest["dialogues_succeeded"] == 0
    assert manifest["dialogues_failed"] == 2
    assert manifest["results"][0]["dialogue_id"] is None
    assert manifest["results"][0]["action"] == "rendered"
    assert manifest["results"][0]["error"] == {
        "message": "Input validation failed: malformed input",
        "type": "ValidationError",
    }


def test_config_failure_remains_startup_failure_without_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr(cli, "configure_logging", lambda *_: None)
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _: (_ for _ in ()).throw(ConfigError("invalid config")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["5703tts", "--output", str(output_root), "--config", "invalid.yaml"],
    )

    with pytest.raises(ConfigError, match="invalid config"):
        asyncio.run(cli.main())

    assert not (output_root / "batch_result.json").exists()


@pytest.mark.parametrize("exit_code", [0, 1])
def test_console_wrapper_raises_system_exit(
    exit_code: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_main() -> int:
        return exit_code

    monkeypatch.setattr(cli, "main", fake_main)

    with pytest.raises(SystemExit) as caught:
        cli.run()

    assert caught.value.code == exit_code


def test_normal_mode_does_not_skip_existing_complete_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes = {
        "a.json": _success(tmp_path, "A"),
        "b.json": _success(tmp_path, "B"),
        "c.json": _success(tmp_path, "C"),
    }
    for dialogue_id in ("A", "B", "C"):
        _complete_dialogue_output(tmp_path / "output", dialogue_id)
    previous = _previous_manifest(
        tmp_path,
        [
            _previous_entry(tmp_path, "a.json", "A", "success"),
            _previous_entry(tmp_path, "b.json", "B", "success"),
            _previous_entry(tmp_path, "c.json", "C", "success"),
        ],
        status="success",
    )

    exit_code, manifest, attempted = _run_batch(
        tmp_path,
        monkeypatch,
        outcomes,
        previous_manifest=previous,
        complete_outputs=True,
    )

    assert attempted == ["a.json", "b.json", "c.json"]
    assert exit_code == 0
    assert manifest["resume_requested"] is False
    assert manifest["dialogues_skipped"] == 0
    assert [result["action"] for result in manifest["results"]] == [
        "rendered",
        "rendered",
        "rendered",
    ]


def test_resume_skips_all_verified_completed_dialogues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes = {
        "a.json": _success(tmp_path, "A"),
        "b.json": _success(tmp_path, "B"),
        "c.json": _success(tmp_path, "C"),
    }
    for dialogue_id in ("A", "B", "C"):
        _complete_dialogue_output(tmp_path / "output", dialogue_id)
    previous = _previous_manifest(
        tmp_path,
        [
            _previous_entry(tmp_path, "a.json", "A", "success"),
            _previous_entry(tmp_path, "b.json", "B", "success"),
            _previous_entry(tmp_path, "c.json", "C", "success"),
        ],
        status="success",
    )

    exit_code, manifest, attempted = _run_batch(
        tmp_path,
        monkeypatch,
        outcomes,
        extra_args=["--resume"],
        previous_manifest=previous,
    )

    assert attempted == []
    assert exit_code == 0
    assert manifest["status"] == "success"
    assert manifest["resume_requested"] is True
    assert manifest["dialogues_total"] == 3
    assert manifest["dialogues_succeeded"] == 3
    assert manifest["dialogues_failed"] == 0
    assert manifest["dialogues_rendered"] == 0
    assert manifest["dialogues_retried"] == 0
    assert manifest["dialogues_skipped"] == 3
    assert [result["action"] for result in manifest["results"]] == [
        "skipped_completed",
        "skipped_completed",
        "skipped_completed",
    ]
    assert [result["status"] for result in manifest["results"]] == [
        "success",
        "success",
        "success",
    ]


def test_resume_retries_previous_failure_and_skips_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_outcomes = {
        "a.json": _success(tmp_path, "A"),
        "b.json": _failed("B", "Unexpected error: worker vanished", "RuntimeError"),
        "c.json": _success(tmp_path, "C"),
    }
    exit_code, first_manifest, attempted = _run_batch(
        tmp_path, monkeypatch, first_outcomes, complete_outputs=True
    )

    assert attempted == ["a.json", "b.json", "c.json"]
    assert exit_code == 1
    assert first_manifest["status"] == "partial_failure"
    assert (tmp_path / "output" / "A" / "preserved.txt").exists()
    assert (tmp_path / "output" / "C" / "preserved.txt").exists()

    retry_outcomes = {
        "a.json": _failed("A", "should not render A", "RuntimeError"),
        "b.json": _success(tmp_path, "B"),
        "c.json": _failed("C", "should not render C", "RuntimeError"),
    }
    exit_code, manifest, attempted = _run_batch(
        tmp_path,
        monkeypatch,
        retry_outcomes,
        extra_args=["--resume"],
        write_inputs=False,
        complete_outputs=True,
    )

    assert attempted == ["b.json"]
    assert exit_code == 0
    assert manifest["status"] == "success"
    assert manifest["resume_requested"] is True
    assert manifest["dialogues_total"] == 3
    assert manifest["dialogues_succeeded"] == 3
    assert manifest["dialogues_failed"] == 0
    assert manifest["dialogues_rendered"] == 1
    assert manifest["dialogues_retried"] == 1
    assert manifest["dialogues_skipped"] == 2
    assert [result["action"] for result in manifest["results"]] == [
        "skipped_completed",
        "retried",
        "skipped_completed",
    ]
    assert [result["status"] for result in manifest["results"]] == [
        "success",
        "success",
        "success",
    ]
    assert (tmp_path / "output" / "A" / "preserved.txt").exists()
    assert (tmp_path / "output" / "C" / "preserved.txt").exists()


def test_resume_retry_failure_keeps_partial_failure_and_skipped_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for dialogue_id in ("A", "C"):
        _complete_dialogue_output(tmp_path / "output", dialogue_id)
    previous = _previous_manifest(
        tmp_path,
        [
            _previous_entry(tmp_path, "a.json", "A", "success"),
            _previous_entry(tmp_path, "b.json", "B", "failed"),
            _previous_entry(tmp_path, "c.json", "C", "success"),
        ],
        status="partial_failure",
    )
    outcomes = {
        "a.json": _success(tmp_path, "A"),
        "b.json": _failed("B", "Unexpected error: still broken", "RuntimeError"),
        "c.json": _success(tmp_path, "C"),
    }

    exit_code, manifest, attempted = _run_batch(
        tmp_path,
        monkeypatch,
        outcomes,
        extra_args=["--resume"],
        previous_manifest=previous,
    )

    assert attempted == ["b.json"]
    assert exit_code == 1
    assert manifest["status"] == "partial_failure"
    assert manifest["dialogues_succeeded"] == 2
    assert manifest["dialogues_failed"] == 1
    assert manifest["dialogues_rendered"] == 1
    assert manifest["dialogues_retried"] == 1
    assert manifest["dialogues_skipped"] == 2
    assert manifest["results"][1]["action"] == "retried"
    assert manifest["results"][1]["status"] == "failed"
    assert (tmp_path / "output" / "A" / "A_clean.wav").exists()
    assert (tmp_path / "output" / "C" / "C_clean.wav").exists()


def test_resume_rerenders_when_required_artifact_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = _complete_dialogue_output(tmp_path / "output", "A")
    (out_dir / "A_telephone.wav").unlink()
    previous = _previous_manifest(
        tmp_path,
        [_previous_entry(tmp_path, "a.json", "A", "success")],
        status="success",
    )
    outcomes = {"a.json": _success(tmp_path, "A")}

    exit_code, manifest, attempted = _run_batch(
        tmp_path,
        monkeypatch,
        outcomes,
        extra_args=["--resume"],
        previous_manifest=previous,
        complete_outputs=True,
    )

    assert attempted == ["a.json"]
    assert exit_code == 0
    assert manifest["results"][0]["action"] == "retried"
    assert manifest["dialogues_skipped"] == 0
    assert manifest["dialogues_retried"] == 1


def test_resume_does_not_skip_when_only_output_directory_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "output" / "A").mkdir(parents=True)
    previous = _previous_manifest(
        tmp_path,
        [_previous_entry(tmp_path, "a.json", "A", "success")],
        status="success",
    )
    outcomes = {"a.json": _success(tmp_path, "A")}

    _, manifest, attempted = _run_batch(
        tmp_path,
        monkeypatch,
        outcomes,
        extra_args=["--resume"],
        previous_manifest=previous,
        complete_outputs=True,
    )

    assert attempted == ["a.json"]
    assert manifest["results"][0]["action"] == "retried"


def test_resume_rerenders_when_input_hash_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _complete_dialogue_output(tmp_path / "output", "A")
    previous = _previous_manifest(
        tmp_path,
        [_previous_entry(tmp_path, "a.json", "A", "success")],
        status="success",
    )
    outcomes = {"a.json": _success(tmp_path, "A")}

    _, manifest, attempted = _run_batch(
        tmp_path,
        monkeypatch,
        outcomes,
        extra_args=["--resume"],
        previous_manifest=previous,
        input_bodies={"a.json": '{"changed": true}\n'},
        complete_outputs=True,
    )

    assert attempted == ["a.json"]
    assert manifest["results"][0]["action"] == "rendered"
    assert manifest["results"][0]["input_sha256"] == _sha256_text('{"changed": true}\n')
    assert manifest["dialogues_skipped"] == 0


def test_resume_rerenders_when_config_hash_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _complete_dialogue_output(tmp_path / "output", "A")
    previous = _previous_manifest(
        tmp_path,
        [_previous_entry(tmp_path, "a.json", "A", "success")],
        status="success",
        config_sha256="0" * 64,
    )
    outcomes = {"a.json": _success(tmp_path, "A")}

    _, manifest, attempted = _run_batch(
        tmp_path,
        monkeypatch,
        outcomes,
        extra_args=["--resume"],
        previous_manifest=previous,
        complete_outputs=True,
    )

    assert attempted == ["a.json"]
    assert manifest["results"][0]["action"] == "rendered"
    assert manifest["config_sha256"] == _sha256_text(_CONFIG_TEXT)
    assert manifest["dialogues_skipped"] == 0


def test_resume_old_step8a_manifest_rerenders_and_writes_fingerprints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _complete_dialogue_output(tmp_path / "output", "A")
    previous = _previous_manifest(
        tmp_path,
        [
            _previous_entry(
                tmp_path, "a.json", "A", "success", include_fingerprints=False
            )
        ],
        status="success",
        include_fingerprints=False,
    )
    outcomes = {"a.json": _success(tmp_path, "A")}

    exit_code, manifest, attempted = _run_batch(
        tmp_path,
        monkeypatch,
        outcomes,
        extra_args=["--resume"],
        previous_manifest=previous,
        complete_outputs=True,
    )

    assert attempted == ["a.json"]
    assert exit_code == 0
    assert manifest["results"][0]["action"] == "rendered"
    assert manifest["config_sha256"] == _sha256_text(_CONFIG_TEXT)
    assert manifest["results"][0]["input_sha256"] == _sha256_text(_INPUT_BODY)
    assert manifest["backend_identity"] == {"backend": "edge_tts"}
    assert manifest["dialogues_skipped"] == 0


def test_resume_without_previous_manifest_renders_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes = {
        "a.json": _success(tmp_path, "A"),
        "b.json": _success(tmp_path, "B"),
    }

    exit_code, manifest, attempted = _run_batch(
        tmp_path,
        monkeypatch,
        outcomes,
        extra_args=["--resume"],
        complete_outputs=True,
    )

    assert attempted == ["a.json", "b.json"]
    assert exit_code == 0
    assert manifest["resume_requested"] is True
    assert manifest["status"] == "success"
    assert manifest["dialogues_skipped"] == 0
    assert [result["action"] for result in manifest["results"]] == [
        "rendered",
        "rendered",
    ]


def test_resume_malformed_manifest_fails_before_render_and_preserves_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    input_root.mkdir()
    output_root.mkdir()
    config_path.write_text(_CONFIG_TEXT, encoding="utf-8")
    (input_root / "a.json").write_text(_INPUT_BODY, encoding="utf-8")
    malformed = "{not json"
    manifest_path = output_root / "batch_result.json"
    manifest_path.write_text(malformed, encoding="utf-8")
    attempted: list[str] = []

    async def fake_run_dialogue(
        path: Path, config: dict, output: Path
    ) -> PipelineResult:
        attempted.append(path.name)
        return _success(tmp_path, "A")

    monkeypatch.setattr(cli, "load_config", lambda _: _CONFIG)
    monkeypatch.setattr(cli, "run_dialogue", fake_run_dialogue)
    monkeypatch.setattr(cli, "configure_logging", lambda *_: None)
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
            "--resume",
        ],
    )

    with pytest.raises(RuntimeError, match="unreadable batch result") as caught:
        asyncio.run(cli.main())

    assert "run without --resume" in str(caught.value)
    assert attempted == []
    assert manifest_path.read_text(encoding="utf-8") == malformed


def test_verified_completed_result_requires_complete_compatible_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "a.json"
    input_path.write_text(_INPUT_BODY, encoding="utf-8")
    out_dir = _complete_dialogue_output(tmp_path / "output", "A")
    previous = {
        "dialogue_id": "A",
        "status": "success",
        "output_dir": str(out_dir),
    }
    monkeypatch.setattr(cli, "load_and_validate", lambda *_: _one_turn_dialogue("A"))

    result = cli._verified_completed_result(
        input_path, previous, _CONFIG, tmp_path / "output"
    )
    assert result is not None
    assert result.status == "success"
    assert result.dialogue_id == "A"

    (out_dir / "A_telephone.wav").unlink()
    assert (
        cli._verified_completed_result(
            input_path, previous, _CONFIG, tmp_path / "output"
        )
        is None
    )

    _complete_dialogue_output(tmp_path / "output", "A")
    (out_dir / "A_metadata.json").unlink()
    assert (
        cli._verified_completed_result(
            input_path, previous, _CONFIG, tmp_path / "output"
        )
        is None
    )

    for child in out_dir.iterdir():
        child.unlink()
    assert (
        cli._verified_completed_result(
            input_path, previous, _CONFIG, tmp_path / "output"
        )
        is None
    )


def test_duplicate_dialogue_ids_are_rejected_before_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes = {
        "a.json": _success(tmp_path, "d001"),
        "b.json": _success(tmp_path, "d001"),
    }
    attempted: list[str] = []

    with pytest.raises(RuntimeError, match="Duplicate dialogue_id 'd001'") as caught:
        _run_batch(tmp_path, monkeypatch, outcomes, attempted=attempted)

    message = str(caught.value)
    assert str(tmp_path / "input" / "a.json") in message
    assert str(tmp_path / "input" / "b.json") in message
    assert attempted == []
    assert not (tmp_path / "output" / "batch_result.json").exists()
    assert not (tmp_path / "output" / "d001").exists()


def test_duplicate_after_unique_ids_still_renders_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes = {
        "a.json": _success(tmp_path, "d001"),
        "b.json": _success(tmp_path, "d002"),
        "c.json": _success(tmp_path, "d001"),
    }
    attempted: list[str] = []

    with pytest.raises(RuntimeError, match="Duplicate dialogue_id 'd001'"):
        _run_batch(tmp_path, monkeypatch, outcomes, attempted=attempted)

    assert attempted == []
    assert not (tmp_path / "output" / "batch_result.json").exists()
    assert not (tmp_path / "output" / "d001").exists()
    assert not (tmp_path / "output" / "d002").exists()


def test_multiple_duplicate_dialogue_id_groups_are_all_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes = {
        "a.json": _success(tmp_path, "d001"),
        "b.json": _success(tmp_path, "d001"),
        "c.json": _success(tmp_path, "d002"),
        "d.json": _success(tmp_path, "d002"),
        "e.json": _success(tmp_path, "d003"),
    }
    attempted: list[str] = []

    with pytest.raises(RuntimeError, match="Duplicate dialogue_id") as caught:
        _run_batch(tmp_path, monkeypatch, outcomes, attempted=attempted)

    message = str(caught.value)
    assert "Duplicate dialogue_id 'd001' found in:" in message
    assert "Duplicate dialogue_id 'd002' found in:" in message
    assert message.index("d001") < message.index("d002")
    assert str(tmp_path / "input" / "a.json") in message
    assert str(tmp_path / "input" / "b.json") in message
    assert str(tmp_path / "input" / "c.json") in message
    assert str(tmp_path / "input" / "d.json") in message
    assert "d003" not in message
    assert attempted == []


def test_unique_dialogue_ids_reach_the_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes = {
        "a.json": _success(tmp_path, "d001"),
        "copy_a.json": _success(tmp_path, "d002"),
        "c.json": _success(tmp_path, "d003"),
    }

    exit_code, manifest, attempted = _run_batch(tmp_path, monkeypatch, outcomes)

    assert attempted == ["a.json", "c.json", "copy_a.json"]
    assert exit_code == 0
    assert manifest["status"] == "success"
    assert [result["dialogue_id"] for result in manifest["results"]] == [
        "d001",
        "d003",
        "d002",
    ]
    assert manifest["dialogues_total"] == 3
    assert manifest["dialogues_failed"] == 0


def test_resume_rejects_current_duplicate_ids_before_skip_or_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = _complete_dialogue_output(tmp_path / "output", "d001")
    marker = out_dir / "preserved.txt"
    marker.write_text("keep\n", encoding="utf-8")
    previous = _previous_manifest(
        tmp_path,
        [_previous_entry(tmp_path, "a.json", "d001", "success")],
        status="success",
    )
    outcomes = {
        "a.json": _success(tmp_path, "d001"),
        "b.json": _success(tmp_path, "d001"),
    }
    attempted: list[str] = []

    with pytest.raises(RuntimeError, match="Duplicate dialogue_id 'd001'"):
        _run_batch(
            tmp_path,
            monkeypatch,
            outcomes,
            extra_args=["--resume"],
            previous_manifest=previous,
            attempted=attempted,
        )

    assert attempted == []
    assert marker.read_text(encoding="utf-8") == "keep\n"
    assert (out_dir / "d001_clean.wav").exists()
    written = json.loads(
        (tmp_path / "output" / "batch_result.json").read_text(encoding="utf-8")
    )
    assert written["dialogues_total"] == 1
    assert written["results"][0]["dialogue_id"] == "d001"


def test_malformed_input_does_not_invent_an_id_and_still_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes = {
        "a.json": _failed(
            None, "Input validation failed: malformed input", "ValidationError"
        ),
        "b.json": _success(tmp_path, "d002"),
    }

    def validate(path: Path, config: dict) -> NormalizedDialogue:
        if path.name == "a.json":
            raise ValidationError("malformed input")
        return _one_turn_dialogue("d002")

    exit_code, manifest, attempted = _run_batch(
        tmp_path, monkeypatch, outcomes, validate=validate
    )

    assert attempted == ["a.json", "b.json"]
    assert exit_code == 1
    assert manifest["status"] == "partial_failure"
    assert manifest["results"][0]["dialogue_id"] is None
    assert manifest["results"][1]["dialogue_id"] == "d002"
