"""Offline tests for the speaker-pool smoke runner.

No real CosyVoice worker is started, no model or GPU is loaded, and no network
is used: synthesis is faked either at the ``synthesize_turn`` boundary or at the
worker protocol boundary inside the production engine adapter.
"""

import csv
import importlib.util
import json
import wave
from functools import cache
from pathlib import Path
from typing import Any

import pytest
import yaml

from tts5703 import tts_engine
from tts5703.validate import NormalizedTurn

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_speaker_pool_smoke.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("speaker_pool_smoke", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
smoke = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(smoke)

SPEAKER_IDS = ("spk_001", "spk_002", "spk_003")
# The real shortlist is twelve speakers; a smaller pool keeps most tests terse.
SHORTLIST_SPEAKER_IDS = tuple(f"spk_{number:03d}" for number in range(1, 13))
REGISTRY_METADATA = {
    "spk_001": {"age": "23", "gender": "F", "accent": "Welsh", "region": "Cardiff"},
    "spk_002": {"age": "26", "gender": "M", "accent": "Indian"},
    "spk_003": {"age": "22", "gender": "M", "accent": "Scottish", "region": "Fife"},
}
DEFAULT_METADATA = {"age": "22", "gender": "M", "accent": "English"}


def _write_wav(path: Path, duration_sec: float, *, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\0\0" * round(duration_sec * sample_rate))
    return path


class Workspace:
    """One synthetic speaker-pool workspace with a materialised-looking registry."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.registry_path = root / "pool" / "speaker_registry.json"
        self.config_path = root / "config.yaml"
        self.output_root = root / "smoke_runs"

    def run(self, **kwargs: Any) -> tuple[dict[str, Any], Path]:
        options: dict[str, Any] = {
            "config_path": self.config_path,
            "registry_path": self.registry_path,
            "output_root": self.output_root,
            "reference_root": self.root,
            "run_id": "cosyvoice3_20260903T000000Z",
        }
        options.update(kwargs)
        return smoke.run_speaker_pool_smoke(**options)

    def registry(self) -> dict[str, Any]:
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def write_registry(self, registry: dict[str, Any]) -> None:
        self.registry_path.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def _reference(root: Path, speaker_id: str, role: str, index: int) -> dict[str, Any]:
    relative = f"pool/references/{speaker_id}_{role}.wav"
    _write_wav(root / relative, 4.0 + index * 0.25, sample_rate=16_000)
    source_speaker_id = f"p{100 + index}"
    return {
        "source_utterance_id": (
            f"{source_speaker_id}_{'001' if role == 'primary' else '002'}"
        ),
        "source_audio_path": f"wav48_silence_trimmed/{source_speaker_id}/file.flac",
        "source_transcript_path": f"txt/{source_speaker_id}/file.txt",
        "prompt_wav": relative,
        # Punctuation and an apostrophe make byte-exact preservation observable.
        "prompt_text": (
            f"It's the {role} reference transcript for {speaker_id}, kept verbatim."
        ),
        "original_duration_sec": 4.0,
        "materialised_duration_sec": 4.0,
        "original_sample_rate": 48_000,
        "materialised_sample_rate": 16_000,
        "sha256": "0" * 64,
    }


def _build_workspace(
    root: Path,
    *,
    engine: str = "cosyvoice",
    speaker_ids: tuple[str, ...] = SPEAKER_IDS,
) -> Workspace:
    workspace = Workspace(root)
    workspace.registry_path.parent.mkdir(parents=True, exist_ok=True)
    workspace.write_registry(
        {
            "speaker_pool_version": "vctk_v0.1",
            "status": "provisional_smoke_test",
            "speakers": [
                {
                    "speaker_id": speaker_id,
                    "source_speaker_id": f"p{100 + index}",
                    "source_metadata": REGISTRY_METADATA.get(
                        speaker_id, DEFAULT_METADATA
                    ),
                    "primary_reference": _reference(root, speaker_id, "primary", index),
                    "backup_reference": _reference(root, speaker_id, "backup", index),
                }
                for index, speaker_id in enumerate(speaker_ids, start=1)
            ],
        }
    )

    cosyvoice_root = root / "fake_cosyvoice"
    (cosyvoice_root / "repo").mkdir(parents=True, exist_ok=True)
    (cosyvoice_root / "model").mkdir(parents=True, exist_ok=True)
    (cosyvoice_root / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    production_prompt = _write_wav(
        cosyvoice_root / "production_prompt.wav", 3.0, sample_rate=16_000
    )
    workspace.config_path.write_text(
        yaml.safe_dump(
            {
                "speaker_voice_map": {"counsellor": "en-US-AriaNeural"},
                "tts": {
                    "engine": engine,
                    "default_rate": "+0%",
                    "kokoro": {
                        "lang_code": "a",
                        "sample_rate": 24_000,
                        "device": None,
                        "voice_map": {"counsellor": "af_heart"},
                    },
                    "cosyvoice": {
                        "python_bin": str(cosyvoice_root / "python"),
                        "repo_dir": str(cosyvoice_root / "repo"),
                        "model_dir": str(cosyvoice_root / "model"),
                        "fp16": True,
                        "load_trt": False,
                        "load_vllm": False,
                        "voice_map": {
                            "counsellor": {
                                "prompt_wav": str(production_prompt),
                                "prompt_text": "Production reference transcript.",
                            }
                        },
                    },
                },
                "pause": {"default_ms": 500},
                "fade_ms": 5,
                "telephone": {
                    "sample_rate": 8_000,
                    "channels": 1,
                    "high_pass_hz": 300,
                    "low_pass_hz": 3_400,
                    "volume_db_reduction": 3,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return _build_workspace(tmp_path / "workspace")


class RecordingSynthesizer:
    """Fake stand-in for the production coroutine, recording every request."""

    def __init__(self, *, failing_prompts: tuple[str, ...] = ()) -> None:
        self.calls: list[dict[str, Any]] = []
        self.failing_prompts = failing_prompts

    async def __call__(
        self, turn: NormalizedTurn, out_dir: Path, config: dict[str, Any]
    ) -> Path:
        voice_map = config["tts"]["cosyvoice"]["voice_map"]
        voice = voice_map[smoke.SMOKE_SPEAKER_ROLE]
        self.calls.append(
            {
                "turn": turn,
                "out_dir": out_dir,
                "voice": voice,
                "voice_map_roles": list(voice_map),
            }
        )
        if Path(voice["prompt_wav"]).stem in self.failing_prompts:
            raise RuntimeError("synthetic synthesis failure")
        return _write_wav(
            out_dir / f"turn_{turn.turn_id:03d}.wav", 2.5, sample_rate=24_000
        )


@pytest.fixture
def synthesizer(monkeypatch: pytest.MonkeyPatch) -> RecordingSynthesizer:
    fake = RecordingSynthesizer()
    monkeypatch.setattr(smoke, "synthesize_turn", fake)
    return fake


def _failing_synthesizer(
    monkeypatch: pytest.MonkeyPatch, *stems: str
) -> RecordingSynthesizer:
    fake = RecordingSynthesizer(failing_prompts=stems)
    monkeypatch.setattr(smoke, "synthesize_turn", fake)
    return fake


def test_exactly_the_registry_speakers_are_attempted(
    workspace: Workspace, synthesizer: RecordingSynthesizer
) -> None:
    results, run_dir = workspace.run()

    assert results["expected_speaker_ids"] == list(SPEAKER_IDS)
    assert [result["speaker_id"] for result in results["speaker_results"]] == list(
        SPEAKER_IDS
    )
    assert len(synthesizer.calls) == len(SPEAKER_IDS)
    assert results["status"] == "complete"
    assert sorted(path.name for path in run_dir.iterdir()) == [
        "smoke_results.json",
        "smoke_review.tsv",
        *SPEAKER_IDS,
    ]


def test_every_speaker_receives_exactly_the_same_text(
    workspace: Workspace, synthesizer: RecordingSynthesizer
) -> None:
    results, _ = workspace.run()

    texts = {call["turn"].text for call in synthesizer.calls}
    assert texts == {smoke.DEFAULT_SMOKE_TEXT}
    assert results["test_sentence"] == smoke.DEFAULT_SMOKE_TEXT
    assert smoke.DEFAULT_SMOKE_TEXT == (
        "I understand. Take your time and tell me what happened when you are ready."
    )


def test_a_custom_sentence_is_still_shared_by_every_speaker(
    workspace: Workspace, synthesizer: RecordingSynthesizer
) -> None:
    results, _ = workspace.run(text="Please continue whenever you feel able.")

    assert {call["turn"].text for call in synthesizer.calls} == {
        "Please continue whenever you feel able."
    }
    assert results["test_sentence"] == "Please continue whenever you feel able."


def test_requested_acoustic_conditions_are_neutral_for_every_speaker(
    workspace: Workspace, synthesizer: RecordingSynthesizer
) -> None:
    results, _ = workspace.run()

    for call in synthesizer.calls:
        turn = call["turn"]
        assert turn.rate == "normal"
        assert turn.pause_before_ms == 0
        assert turn.pause_after_ms == 0
        assert turn.arousal is None
        assert turn.coarse_affect is None
        assert turn.emotion is None
        assert turn.paralinguistic_events == []
        assert turn.label == "normal"
    assert results["requested_acoustic_conditions"] == {
        "rate": "normal",
        "pause_before_ms": 0,
        "pause_after_ms": 0,
        "arousal": None,
        "coarse_affect": None,
        "emotion": None,
        "paralinguistic_events": [],
    }


def test_runner_calls_the_production_synthesis_entry_point() -> None:
    assert smoke.synthesize_turn is tts_engine.synthesize_turn


def test_primary_reference_is_used_by_default(
    workspace: Workspace, synthesizer: RecordingSynthesizer
) -> None:
    results, _ = workspace.run()

    registry = workspace.registry()
    for entry, result in zip(
        registry["speakers"], results["speaker_results"], strict=True
    ):
        expected = entry["primary_reference"]
        assert result["prompt_wav"] == expected["prompt_wav"]
        assert result["prompt_text"] == expected["prompt_text"]
        assert result["source_utterance_id"] == expected["source_utterance_id"]
    assert results["reference_role"] == "primary"
    assert all("_primary" in call["voice"]["prompt_wav"] for call in synthesizer.calls)


def test_backup_reference_can_be_requested(
    workspace: Workspace, synthesizer: RecordingSynthesizer
) -> None:
    results, _ = workspace.run(role="backup")

    assert results["reference_role"] == "backup"
    assert all("_backup" in call["voice"]["prompt_wav"] for call in synthesizer.calls)


def test_zero_shot_prompt_prefix_matches_the_cosyvoice3_example() -> None:
    assert smoke.COSYVOICE3_END_OF_PROMPT == "<|endofprompt|>"
    assert smoke.COSYVOICE3_ZERO_SHOT_PROMPT_PREFIX == (
        "You are a helpful assistant.<|endofprompt|>"
    )
    assert smoke.format_cosyvoice3_zero_shot_prompt_text("Hello there.") == (
        "You are a helpful assistant.<|endofprompt|>Hello there."
    )


@pytest.mark.parametrize(
    "raw_transcript", ["", "   ", "\n"], ids=["empty", "spaces", "newline"]
)
def test_empty_raw_transcript_is_rejected(raw_transcript: str) -> None:
    with pytest.raises(smoke.SpeakerSmokeError, match="non-empty"):
        smoke.format_cosyvoice3_zero_shot_prompt_text(raw_transcript)


@pytest.mark.parametrize(
    "raw_transcript",
    [
        "You are a helpful assistant.<|endofprompt|>Hello there.",
        "Hello <|endofprompt|> there.",
    ],
    ids=["already_prefixed", "marker_inside"],
)
def test_already_marked_transcript_is_rejected_instead_of_double_prefixed(
    raw_transcript: str,
) -> None:
    with pytest.raises(smoke.SpeakerSmokeError) as error:
        smoke.format_cosyvoice3_zero_shot_prompt_text(raw_transcript)

    message = str(error.value)
    assert "double-prefix" in message
    assert "<|endofprompt|>" in message


def test_model_facing_config_receives_the_formatted_prompt_text(
    workspace: Workspace, synthesizer: RecordingSynthesizer
) -> None:
    results, _ = workspace.run()

    registry = workspace.registry()
    for entry, call, result in zip(
        registry["speakers"], synthesizer.calls, results["speaker_results"], strict=True
    ):
        raw = entry["primary_reference"]["prompt_text"]
        model_facing = call["voice"]["prompt_text"]
        assert model_facing == f"You are a helpful assistant.<|endofprompt|>{raw}"
        # The registry transcript itself must reach the model unchanged.
        assert model_facing.split("<|endofprompt|>", 1)[1] == raw
        assert model_facing.count("<|endofprompt|>") == 1
        # The report keeps the raw transcript plus a trace of the formatting.
        assert result["prompt_text"] == raw
        assert "<|endofprompt|>" not in result["prompt_text"]
        assert result["prompt_format"] == "cosyvoice3_zero_shot"
    assert results["prompt_formatting"] == {
        "prompt_format": "cosyvoice3_zero_shot",
        "model_facing_prefix": "You are a helpful assistant.<|endofprompt|>",
        "applied_to": "model_facing_prompt_text_only",
        "note": results["prompt_formatting"]["note"],
    }


def test_all_twelve_shortlist_speakers_receive_the_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _build_workspace(
        tmp_path / "shortlist", speaker_ids=SHORTLIST_SPEAKER_IDS
    )
    recorder = RecordingSynthesizer()
    monkeypatch.setattr(smoke, "synthesize_turn", recorder)

    results, _ = workspace.run()

    assert results["status"] == "complete"
    assert len(recorder.calls) == 12
    registry = workspace.registry()
    for entry, call in zip(registry["speakers"], recorder.calls, strict=True):
        raw = entry["primary_reference"]["prompt_text"]
        assert call["voice"]["prompt_text"] == (
            f"You are a helpful assistant.<|endofprompt|>{raw}"
        )
    assert all(
        result["prompt_format"] == "cosyvoice3_zero_shot"
        for result in results["speaker_results"]
    )


def test_registry_prompt_text_is_never_rewritten(
    workspace: Workspace, synthesizer: RecordingSynthesizer
) -> None:
    before = workspace.registry_path.read_bytes()

    workspace.run()

    assert workspace.registry_path.read_bytes() == before
    for entry in workspace.registry()["speakers"]:
        for role in ("primary", "backup"):
            raw = entry[f"{role}_reference"]["prompt_text"]
            assert "<|endofprompt|>" not in raw
            assert "helpful assistant" not in raw
            assert raw.startswith("It's the ")


def test_an_already_marked_registry_transcript_fails_only_that_speaker(
    workspace: Workspace, synthesizer: RecordingSynthesizer
) -> None:
    registry = workspace.registry()
    registry["speakers"][1]["primary_reference"]["prompt_text"] = (
        "You are a helpful assistant.<|endofprompt|>Already formatted."
    )
    workspace.write_registry(registry)

    results, _ = workspace.run()

    assert [result["speaker_id"] for result in results["speaker_results"]] == list(
        SPEAKER_IDS
    )
    failed = results["speaker_results"][1]
    assert failed["success"] is False
    assert failed["prompt_format"] is None
    assert "double-prefix" in failed["error"]
    assert results["status"] == "partial_failure"
    assert len(synthesizer.calls) == 2
    assert all(
        call["voice"]["prompt_text"].count("<|endofprompt|>") == 1
        for call in synthesizer.calls
    )


def test_config_file_and_base_voice_map_are_left_untouched(
    workspace: Workspace, synthesizer: RecordingSynthesizer
) -> None:
    before = workspace.config_path.read_bytes()

    workspace.run()

    assert workspace.config_path.read_bytes() == before
    config = yaml.safe_load(before.decode("utf-8"))
    assert list(config["tts"]["cosyvoice"]["voice_map"]) == ["counsellor"]
    # The temporary smoke role exists only in the per-speaker in-memory copies.
    for call in synthesizer.calls:
        assert call["voice_map_roles"] == ["counsellor", smoke.SMOKE_SPEAKER_ROLE]


def test_non_cosyvoice_engine_is_rejected(tmp_path: Path) -> None:
    workspace = _build_workspace(tmp_path / "kokoro", engine="kokoro")

    with pytest.raises(smoke.SpeakerSmokeError, match="cosyvoice"):
        workspace.run()


def test_missing_registry_is_reported_with_the_next_step(workspace: Workspace) -> None:
    workspace.registry_path.unlink()

    with pytest.raises(
        smoke.SpeakerSmokeError, match="materialize_vctk_speaker_pool.py"
    ):
        workspace.run()


def test_missing_reference_audio_is_recorded_as_a_speaker_failure(
    workspace: Workspace, synthesizer: RecordingSynthesizer
) -> None:
    registry = workspace.registry()
    missing = registry["speakers"][1]["primary_reference"]["prompt_wav"]
    (workspace.root / missing).unlink()

    results, _ = workspace.run()

    failed = results["speaker_results"][1]
    assert failed["speaker_id"] == "spk_002"
    assert failed["success"] is False
    assert "Reference audio is missing" in failed["error"]
    assert results["status"] == "partial_failure"
    assert len(synthesizer.calls) == 2


def test_one_speaker_failure_does_not_disappear_from_the_results(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _failing_synthesizer(monkeypatch, "spk_002_primary")

    results, run_dir = workspace.run()

    written = json.loads((run_dir / smoke.RESULTS_FILENAME).read_text(encoding="utf-8"))
    assert written == results
    assert [result["speaker_id"] for result in written["speaker_results"]] == list(
        SPEAKER_IDS
    )
    failed = written["speaker_results"][1]
    assert failed["success"] is False
    assert "synthetic synthesis failure" in failed["error"]
    assert failed["duration_sec"] is None
    assert failed["real_time_factor"] is None
    assert written["successful_speakers"] == 2
    assert written["failed_speakers"] == 1
    assert [result["success"] for result in written["speaker_results"]] == [
        True,
        False,
        True,
    ]


@pytest.mark.parametrize(
    ("failing", "expected_status"),
    [
        ((), "complete"),
        (("spk_002_primary",), "partial_failure"),
        (
            ("spk_001_primary", "spk_002_primary", "spk_003_primary"),
            "failed",
        ),
    ],
    ids=["complete", "partial_failure", "failed"],
)
def test_overall_status_reflects_partial_failure(
    workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
    failing: tuple[str, ...],
    expected_status: str,
) -> None:
    _failing_synthesizer(monkeypatch, *failing)

    results, run_dir = workspace.run()

    written = json.loads((run_dir / smoke.RESULTS_FILENAME).read_text(encoding="utf-8"))
    assert results["status"] == expected_status
    assert written["status"] == expected_status
    assert len(written["speaker_results"]) == len(SPEAKER_IDS)


def test_successful_results_record_traceable_timing_without_ranking(
    workspace: Workspace, synthesizer: RecordingSynthesizer
) -> None:
    results, run_dir = workspace.run()

    for result in results["speaker_results"]:
        assert result["success"] is True
        assert result["elapsed_sec"] >= 0
        assert result["duration_sec"] == pytest.approx(2.5, abs=1e-3)
        assert result["real_time_factor"] == pytest.approx(
            result["elapsed_sec"] / result["duration_sec"], abs=1e-4
        )
        assert (run_dir / result["generated_audio"]).is_file()
        assert not Path(result["generated_audio"]).is_absolute()
    assert "not ranked" in results["ranking_disclaimer"]
    assert results["engine"] == "cosyvoice"
    assert results["engine_description"]["model"] == "Fun-CosyVoice3-0.5B"


def _review_rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / smoke.REVIEW_FILENAME).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, dialect="excel-tab"))


def test_review_sheet_contains_every_speaker(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _failing_synthesizer(monkeypatch, "spk_002_primary")

    _, run_dir = workspace.run()

    rows = _review_rows(run_dir)
    assert [row["speaker_id"] for row in rows] == list(SPEAKER_IDS)
    assert list(rows[0]) == [
        "speaker_id",
        "source_speaker_id",
        "gender",
        "accent",
        "region",
        "source_utterance_id",
        "generated_audio",
        "naturalness",
        "identity_preservation",
        "distinct_from_others",
        "artifacts",
        "decision",
        "notes",
    ]
    assert rows[0]["gender"] == "F"
    assert rows[1]["region"] == "", "absent VCTK metadata must stay empty"
    assert rows[1]["generated_audio"] == "", "the failed speaker has no audio"


def test_review_sheet_leaves_every_human_judgment_empty(
    workspace: Workspace, synthesizer: RecordingSynthesizer
) -> None:
    _, run_dir = workspace.run()

    for row in _review_rows(run_dir):
        for column in smoke.HUMAN_REVIEW_COLUMNS:
            assert row[column] == ""


def test_run_directories_do_not_overwrite_each_other(
    workspace: Workspace, synthesizer: RecordingSynthesizer
) -> None:
    _, first = workspace.run()
    _, second = workspace.run()

    assert first != second
    assert first.name == "cosyvoice3_20260903T000000Z"
    assert second.name == "cosyvoice3_20260903T000000Z_01"


def test_production_engine_path_reuses_one_mocked_worker(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive the real synthesize_turn with only the worker protocol faked."""
    workers: list[object] = []
    requests: list[dict[str, Any]] = []

    @cache
    def fake_worker(*init: Any) -> object:
        worker = object()
        workers.append(worker)
        return worker

    def fake_request(worker: object, request: dict[str, Any]) -> dict[str, Any]:
        requests.append({"worker": id(worker), **request})
        _write_wav(Path(request["output_path"]), 2.0, sample_rate=24_000)
        return {"status": "ok"}

    monkeypatch.setattr(tts_engine, "_get_cosyvoice_worker", fake_worker)
    monkeypatch.setattr(tts_engine, "_cosyvoice_request", fake_request)

    results, _ = workspace.run()

    assert results["status"] == "complete"
    assert len(requests) == len(SPEAKER_IDS)
    assert len(workers) == 1, "worker start-up parameters must not vary per speaker"
    assert len({request["worker"] for request in requests}) == 1
    registry = workspace.registry()
    for entry, request in zip(registry["speakers"], requests, strict=True):
        reference = entry["primary_reference"]
        assert request["mode"] == "zero_shot"
        assert "instruction" not in request
        assert request["speed"] == 1.0
        assert request["text"] == smoke.DEFAULT_SMOKE_TEXT
        assert request["prompt_text"] == (
            f"You are a helpful assistant.<|endofprompt|>{reference['prompt_text']}"
        )
        assert request["prompt_wav"] == str(workspace.root / reference["prompt_wav"])


def test_omitted_speaker_ids_still_tests_every_registry_speaker(
    workspace: Workspace, synthesizer: RecordingSynthesizer
) -> None:
    results, _ = workspace.run()

    assert results["expected_speaker_ids"] == list(SPEAKER_IDS)
    assert [result["speaker_id"] for result in results["speaker_results"]] == list(
        SPEAKER_IDS
    )
    assert len(synthesizer.calls) == len(SPEAKER_IDS)


def test_speaker_ids_selects_only_requested_speakers_in_order(
    workspace: Workspace, synthesizer: RecordingSynthesizer
) -> None:
    results, run_dir = workspace.run(speaker_ids=["spk_003", "spk_001"])

    assert results["expected_speaker_ids"] == ["spk_003", "spk_001"]
    assert [result["speaker_id"] for result in results["speaker_results"]] == [
        "spk_003",
        "spk_001",
    ]
    assert results["speaker_count"] == 2
    assert len(synthesizer.calls) == 2
    generated = sorted(path.name for path in run_dir.iterdir())
    assert generated == [
        "smoke_results.json",
        "smoke_review.tsv",
        "spk_001",
        "spk_003",
    ]
    prompt_wavs = [call["voice"]["prompt_wav"] for call in synthesizer.calls]
    assert Path(prompt_wavs[0]).stem == "spk_003_primary"
    assert Path(prompt_wavs[1]).stem == "spk_001_primary"


def test_targeted_smoke_still_calls_synthesize_turn_with_neutral_conditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _build_workspace(
        tmp_path / "targeted", speaker_ids=SHORTLIST_SPEAKER_IDS
    )
    recorder = RecordingSynthesizer()
    monkeypatch.setattr(smoke, "synthesize_turn", recorder)

    results, _ = workspace.run(speaker_ids=["spk_002", "spk_005"], role="backup")

    assert len(recorder.calls) == 2
    assert results["test_sentence"] == smoke.DEFAULT_SMOKE_TEXT
    assert results["reference_role"] == "backup"
    assert results["expected_speaker_ids"] == ["spk_002", "spk_005"]
    for call in recorder.calls:
        turn = call["turn"]
        assert turn.text == smoke.DEFAULT_SMOKE_TEXT
        assert turn.rate == "normal"
        assert turn.pause_before_ms == 0
        assert turn.pause_after_ms == 0
        assert turn.arousal is None
        assert turn.coarse_affect is None
        assert turn.emotion is None
        assert turn.paralinguistic_events == []
        assert "_backup" in call["voice"]["prompt_wav"]
    assert results["requested_acoustic_conditions"] == {
        "rate": "normal",
        "pause_before_ms": 0,
        "pause_after_ms": 0,
        "arousal": None,
        "coarse_affect": None,
        "emotion": None,
        "paralinguistic_events": [],
    }


def test_unknown_speaker_id_fails_before_synthesis(
    workspace: Workspace, synthesizer: RecordingSynthesizer
) -> None:
    with pytest.raises(smoke.SpeakerSmokeError, match="Unknown speaker ID") as error:
        workspace.run(speaker_ids=["spk_001", "spk_999"])

    assert "spk_999" in str(error.value)
    assert "spk_001" in str(error.value)
    assert synthesizer.calls == []
    assert not workspace.output_root.exists()


def test_duplicate_speaker_ids_are_rejected(
    workspace: Workspace, synthesizer: RecordingSynthesizer
) -> None:
    with pytest.raises(smoke.SpeakerSmokeError, match="Duplicate speaker ID"):
        workspace.run(speaker_ids=["spk_002", "spk_002"])

    assert synthesizer.calls == []
    assert not workspace.output_root.exists()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("spk_002,spk_005", ["spk_002", "spk_005"]),
        ("spk_002, spk_005", ["spk_002", "spk_005"]),
        (" spk_010 ", ["spk_010"]),
    ],
    ids=["comma", "comma_space", "single"],
)
def test_parse_speaker_ids_preserves_requested_order(
    raw: str, expected: list[str]
) -> None:
    assert smoke.parse_speaker_ids(raw) == expected
    assert smoke.parse_speaker_ids(None) is None


def test_parse_speaker_ids_rejects_duplicates_and_empty_entries() -> None:
    with pytest.raises(smoke.SpeakerSmokeError, match="Duplicate speaker ID"):
        smoke.parse_speaker_ids("spk_002,spk_005,spk_002")
    with pytest.raises(smoke.SpeakerSmokeError, match="empty"):
        smoke.parse_speaker_ids("spk_002,")
    with pytest.raises(smoke.SpeakerSmokeError, match="non-empty"):
        smoke.parse_speaker_ids("   ")


def test_cli_speaker_ids_option_selects_requested_speakers(
    workspace: Workspace, synthesizer: RecordingSynthesizer
) -> None:
    exit_code = smoke.main(
        [
            "--config",
            str(workspace.config_path),
            "--registry",
            str(workspace.registry_path),
            "--output-root",
            str(workspace.output_root),
            "--reference-root",
            str(workspace.root),
            "--speaker-ids",
            "spk_002,spk_001",
            "--reference",
            "backup",
            "--run-id",
            "cli_targeted",
        ]
    )

    assert exit_code == 0
    results = json.loads(
        (workspace.output_root / "cli_targeted" / smoke.RESULTS_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert results["expected_speaker_ids"] == ["spk_002", "spk_001"]
    assert results["reference_role"] == "backup"
    assert [result["speaker_id"] for result in results["speaker_results"]] == [
        "spk_002",
        "spk_001",
    ]
    assert all("_backup" in call["voice"]["prompt_wav"] for call in synthesizer.calls)


def test_cli_unknown_speaker_id_exits_before_synthesis(
    workspace: Workspace,
    synthesizer: RecordingSynthesizer,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = smoke.main(
        [
            "--config",
            str(workspace.config_path),
            "--registry",
            str(workspace.registry_path),
            "--output-root",
            str(workspace.output_root),
            "--reference-root",
            str(workspace.root),
            "--speaker-ids",
            "spk_002,spk_missing",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Unknown speaker ID" in captured.err
    assert "spk_missing" in captured.err
    assert synthesizer.calls == []
    assert not workspace.output_root.exists()


def test_targeted_run_reuses_one_mocked_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _build_workspace(
        tmp_path / "targeted_worker", speaker_ids=SHORTLIST_SPEAKER_IDS
    )
    workers: list[object] = []
    requests: list[dict[str, Any]] = []

    @cache
    def fake_worker(*init: Any) -> object:
        worker = object()
        workers.append(worker)
        return worker

    def fake_request(worker: object, request: dict[str, Any]) -> dict[str, Any]:
        requests.append({"worker": id(worker), **request})
        _write_wav(Path(request["output_path"]), 2.0, sample_rate=24_000)
        return {"status": "ok"}

    monkeypatch.setattr(tts_engine, "_get_cosyvoice_worker", fake_worker)
    monkeypatch.setattr(tts_engine, "_cosyvoice_request", fake_request)

    results, _ = workspace.run(speaker_ids=["spk_002", "spk_005"], role="backup")

    assert results["status"] == "complete"
    assert results["expected_speaker_ids"] == ["spk_002", "spk_005"]
    assert len(requests) == 2
    assert len(workers) == 1
    assert len({request["worker"] for request in requests}) == 1
    registry = {
        entry["speaker_id"]: entry for entry in workspace.registry()["speakers"]
    }
    for speaker_id, request in zip(("spk_002", "spk_005"), requests, strict=True):
        reference = registry[speaker_id]["backup_reference"]
        assert request["mode"] == "zero_shot"
        assert "instruction" not in request
        assert request["text"] == smoke.DEFAULT_SMOKE_TEXT
        assert request["prompt_text"] == (
            f"You are a helpful assistant.<|endofprompt|>{reference['prompt_text']}"
        )
