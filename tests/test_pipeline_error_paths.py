"""Expected versus unexpected failure reporting in run_dialogue; no synthesis."""

import asyncio
import json
import logging
from pathlib import Path

import pytest

from tts5703 import pipeline
from tts5703.config import load_config
from tts5703.tts_engine import BackendControlError

CONFIG_PATH = Path("config/config.yaml")


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config(CONFIG_PATH)


def _dialogue_file(tmp_path: Path, coarse_affect: str | None = None) -> Path:
    path = tmp_path / "dialogue001.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "0.2",
                "dialogue_id": "dialogue001",
                "turns": [
                    {
                        "turn_id": 1,
                        "speaker": "caller",
                        "text": "I do not know what to do next.",
                        "label": "alert",
                        "acoustic_spec": {"coarse_affect": coarse_affect},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _run(json_path: Path, config: dict, tmp_path: Path) -> pipeline.PipelineResult:
    return asyncio.run(pipeline.run_dialogue(json_path, config, tmp_path / "out"))


def test_backend_control_error_is_reported_as_an_expected_failure(
    tmp_path: Path, config: dict, caplog: pytest.LogCaptureFixture
) -> None:
    json_path = _dialogue_file(tmp_path, coarse_affect="anxious")

    with caplog.at_level(logging.DEBUG, logger="tts5703.pipeline"):
        result = _run(json_path, config, tmp_path)

    assert result.status == "failed"
    assert result.dialogue_id == "dialogue001"
    assert "Unexpected error" not in result.error
    assert "Backend control preflight failed" in result.error
    assert "Unsupported CosyVoice coarse_affect mapping: 'anxious'" in result.error
    assert "Currently supported mappings: neutral, distressed." in result.error

    records = [
        record for record in caplog.records if "backend_preflight" in record.message
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].exc_info is None


def test_backend_control_failure_renders_no_audio(tmp_path: Path, config: dict) -> None:
    json_path = _dialogue_file(tmp_path, coarse_affect="anxious")

    result = _run(json_path, config, tmp_path)

    assert result.status == "failed"
    assert result.out_dir is None
    assert not (tmp_path / "out" / "dialogue001").exists()


def test_unexpected_error_still_uses_the_generic_unexpected_path(
    tmp_path: Path,
    config: dict,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    json_path = _dialogue_file(tmp_path)

    async def exploding_synthesis(*args: object, **kwargs: object) -> dict:
        raise RuntimeError("worker vanished")

    monkeypatch.setattr(pipeline, "synthesize_all_turns", exploding_synthesis)

    with caplog.at_level(logging.DEBUG, logger="tts5703.pipeline"):
        result = _run(json_path, config, tmp_path)

    assert result.status == "failed"
    assert result.error == "Unexpected error: worker vanished"

    records = [
        record
        for record in caplog.records
        if record.message.startswith("event=dialogue_pipeline_failed")
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert records[0].exc_info is not None


def test_validation_error_path_is_unchanged(
    tmp_path: Path, config: dict, caplog: pytest.LogCaptureFixture
) -> None:
    json_path = tmp_path / "invalid.json"
    json_path.write_text(
        json.dumps({"schema_version": "0.2", "dialogue_id": "x", "turns": []}),
        encoding="utf-8",
    )

    with caplog.at_level(logging.DEBUG, logger="tts5703.pipeline"):
        result = _run(json_path, config, tmp_path)

    assert result.status == "failed"
    assert result.error.startswith("Input validation failed:")
    assert all(record.exc_info is None for record in caplog.records)


def test_backend_control_error_is_not_caught_as_a_broad_value_error(
    tmp_path: Path, config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain ValueError is a defect, so it must keep the unexpected path."""
    json_path = _dialogue_file(tmp_path)

    async def value_error_synthesis(*args: object, **kwargs: object) -> dict:
        raise ValueError("some unrelated defect")

    monkeypatch.setattr(pipeline, "synthesize_all_turns", value_error_synthesis)

    result = _run(json_path, config, tmp_path)

    assert result.error == "Unexpected error: some unrelated defect"


def test_backend_control_error_subclass_is_still_matched(
    tmp_path: Path, config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    json_path = _dialogue_file(tmp_path)

    async def backend_error_synthesis(*args: object, **kwargs: object) -> dict:
        raise BackendControlError("turn 1: Unsupported CosyVoice arousal mapping")

    monkeypatch.setattr(pipeline, "synthesize_all_turns", backend_error_synthesis)

    result = _run(json_path, config, tmp_path)

    assert result.error == (
        "Backend control preflight failed: "
        "turn 1: Unsupported CosyVoice arousal mapping"
    )
