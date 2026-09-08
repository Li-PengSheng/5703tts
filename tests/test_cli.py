"""Offline batch-result and process-status tests for the production CLI."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

from tts5703 import cli
from tts5703.config import ConfigError
from tts5703.pipeline import PipelineResult


def _run_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcomes: dict[str, PipelineResult],
) -> tuple[int, dict, list[str]]:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    for filename in outcomes:
        (input_root / filename).write_text("{}\n", encoding="utf-8")

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
        return result

    monkeypatch.setattr(cli, "load_config", lambda _: {"tts": {"engine": "fake"}})
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
            str(tmp_path / "config.yaml"),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    exit_code = asyncio.run(cli.main())
    manifest = json.loads(
        (output_root / "batch_result.json").read_text(encoding="utf-8")
    )
    assert not (output_root / ".batch_result.json.tmp").exists()
    return exit_code, manifest, attempted


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
    assert [result["status"] for result in manifest["results"]] == [
        "success",
        "success",
    ]
    assert [result["output_dir"] for result in manifest["results"]] == [
        str(output / "A"),
        str(output / "B"),
    ]
    assert all("error" not in result for result in manifest["results"])


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
        "dialogue_id": "B",
        "error": {
            "message": "Unexpected error: worker vanished",
            "type": "RuntimeError",
        },
        "input_path": str(tmp_path / "input" / "b.json"),
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
