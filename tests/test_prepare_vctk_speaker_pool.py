"""Deterministic offline tests for the VCTK candidate indexer."""

import csv
import importlib.util
import json
import wave
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "prepare_vctk_speaker_pool.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("vctk_pool", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
vctk_pool = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(vctk_pool)


def _write_wav(
    path: Path, duration_sec: float, *, sample_rate: int = 8_000, channels: int = 1
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = round(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\0\0" * frame_count * channels)
    return path


def _write_transcript(
    root: Path, speaker_id: str, utterance_id: str, text: str
) -> None:
    path = root / "txt" / speaker_id / f"{utterance_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_tiny_vctk(root: Path) -> Path:
    (root / "wav48_silence_trimmed").mkdir(parents=True)
    (root / "txt").mkdir()
    (root / "speaker-info.txt").write_text(
        "ID  AGE  GENDER  ACCENTS  REGION COMMENTS\n"
        "p225  23  F  English  Southern England\n"
        "p226  41  M  Scottish  Fife\n"
        "p280  25  F  Unknown  France (mic2 files unavailable)\n"
        "p315  18  M  American  New England (Text unavailable)\n",
        encoding="utf-8",
    )

    audio_dir = root / "wav48_silence_trimmed" / "p225"
    valid_text = (
        "These twelve clear words make a useful stable reference prompt for testing."
    )
    ranked_text = (
        "Another clear sentence supplies exactly enough useful words for ranking tests."
    )
    for utterance_id, duration, transcript in (
        ("p225_001", 5.0, valid_text),
        (
            "p225_002",
            4.0,
            "This sentence also has enough words to become one candidate.",
        ),
        (
            "p225_003",
            9.0,
            "This sentence has enough words but its recording is much too long.",
        ),
        ("p225_004", 5.0, " \n\t "),
        ("p225_005", 5.0, None),
        ("p225_007", 5.0, ranked_text),
    ):
        _write_wav(audio_dir / f"{utterance_id}_mic1.wav", duration)
        if transcript is not None:
            _write_transcript(root, "p225", utterance_id, transcript)

    _write_wav(audio_dir / "p225_006_mic2.wav", 5.0)
    _write_transcript(
        root,
        "p225",
        "p225_006",
        "This transcript belongs only to the microphone two recording here.",
    )
    (root / "wav48_silence_trimmed" / "p226").mkdir()
    (root / "txt" / "p226").mkdir()

    # Excluded-speaker files prove the indexer does not accidentally include them.
    for speaker_id in ("p280", "p315"):
        utterance_id = f"{speaker_id}_001"
        _write_wav(
            root / "wav48_silence_trimmed" / speaker_id / f"{utterance_id}_mic1.wav",
            5.0,
        )
        _write_transcript(
            root,
            speaker_id,
            utterance_id,
            "Excluded speaker content must never become a reported candidate.",
        )
    return root


@pytest.fixture
def tiny_vctk(tmp_path: Path) -> Path:
    return _build_tiny_vctk(tmp_path / "tiny-vctk")


def _report(tiny_vctk: Path, *, top_k: int = 5) -> dict[str, Any]:
    report, _ = vctk_pool.build_candidate_report(tiny_vctk, top_k=top_k)
    return report


def _speaker(report: dict[str, Any], speaker_id: str = "p225") -> dict[str, Any]:
    return next(
        speaker
        for speaker in report["speakers"]
        if speaker["source_speaker_id"] == speaker_id
    )


def test_source_structure_validation_reports_all_required_paths(tmp_path: Path) -> None:
    with pytest.raises(vctk_pool.VCTKPreparationError) as error:
        vctk_pool.validate_source_structure(tmp_path / "missing-vctk")
    message = str(error.value)
    assert "speaker-info.txt" in message
    assert "txt" in message
    assert "wav48_silence_trimmed" in message
    assert "--vctk-root" in message


def test_speaker_info_parsing_preserves_source_metadata(tiny_vctk: Path) -> None:
    speakers, fields = vctk_pool.parse_speaker_info(tiny_vctk / "speaker-info.txt")
    assert fields == [
        "source_speaker_id",
        "age",
        "gender",
        "accent",
        "region",
        "comments",
    ]
    assert speakers[0] == {
        "source_speaker_id": "p225",
        "age": "23",
        "gender": "F",
        "accent": "English",
        "region": "Southern England",
    }
    assert speakers[2]["comments"] == "(mic2 files unavailable)"


def test_p280_is_excluded_with_a_reason(tiny_vctk: Path) -> None:
    report = _report(tiny_vctk)
    assert "p280" not in {
        speaker["source_speaker_id"] for speaker in report["speakers"]
    }
    excluded = report["source_dataset"]["excluded_speakers"]
    assert any(
        item["source_speaker_id"] == "p280" and item["reason"] for item in excluded
    )


def test_p315_is_excluded_with_a_reason(tiny_vctk: Path) -> None:
    report = _report(tiny_vctk)
    assert "p315" not in {
        speaker["source_speaker_id"] for speaker in report["speakers"]
    }
    excluded = report["source_dataset"]["excluded_speakers"]
    assert any(
        item["source_speaker_id"] == "p315" and item["reason"] for item in excluded
    )


def test_requested_microphone_is_used_consistently(tiny_vctk: Path) -> None:
    report = _report(tiny_vctk)
    speaker = _speaker(report)
    assert speaker["index_counts"]["other_mic_audio_skipped"] == 1
    assert all("_mic1.wav" in item["audio_path"] for item in speaker["candidates"])
    assert all(
        item["source_utterance_id"] != "p225_006" for item in speaker["candidates"]
    )


def test_audio_and_transcript_ids_are_matched(tiny_vctk: Path) -> None:
    candidate = _speaker(_report(tiny_vctk))["candidates"][0]
    assert candidate["source_utterance_id"] == "p225_001"
    assert Path(candidate["audio_path"]).stem == "p225_001_mic1"
    assert Path(candidate["transcript_path"]).stem == "p225_001"
    assert candidate["sample_rate"] == 8_000
    assert candidate["channels"] == 1


def test_missing_transcript_is_skipped_without_crashing(tiny_vctk: Path) -> None:
    speaker = _speaker(_report(tiny_vctk))
    assert speaker["index_counts"]["missing_transcript"] == 1
    assert all(
        item["source_utterance_id"] != "p225_005" for item in speaker["candidates"]
    )


def test_duration_filter_rejects_out_of_range_audio(tiny_vctk: Path) -> None:
    speaker = _speaker(_report(tiny_vctk))
    assert speaker["index_counts"]["duration_outside_range"] == 1
    assert all(
        item["source_utterance_id"] != "p225_003" for item in speaker["candidates"]
    )


def test_empty_transcript_is_rejected(tiny_vctk: Path) -> None:
    speaker = _speaker(_report(tiny_vctk))
    assert speaker["index_counts"]["empty_transcript"] == 1
    assert all(
        item["source_utterance_id"] != "p225_004" for item in speaker["candidates"]
    )


def test_ranking_is_deterministic(tiny_vctk: Path) -> None:
    first = _speaker(_report(tiny_vctk))["candidates"]
    second = _speaker(_report(tiny_vctk))["candidates"]
    assert first == second
    assert [item["source_utterance_id"] for item in first] == [
        "p225_001",
        "p225_007",
        "p225_002",
    ]
    assert [item["rank"] for item in first] == [1, 2, 3]


def test_top_k_is_respected_without_hiding_candidate_count(tiny_vctk: Path) -> None:
    speaker = _speaker(_report(tiny_vctk, top_k=2))
    assert speaker["candidate_count"] == 3
    assert len(speaker["candidates"]) == 2
    assert speaker["index_counts"]["reported_candidates"] == 2


def test_json_output_has_required_structure(tiny_vctk: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "reports"
    report, outputs = vctk_pool.prepare_speaker_pool(tiny_vctk, output_dir, top_k=2)
    loaded = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert loaded == report
    assert set(loaded) == {
        "speaker_pool_version",
        "source_dataset",
        "selection_config",
        "summary",
        "speakers",
    }
    assert loaded["speaker_pool_version"] == "vctk_v0.1"
    assert loaded["summary"]["speakers_indexed"] == 4
    assert loaded["summary"]["speakers_eligible"] == 2


def test_tsv_output_has_one_row_per_reported_candidate(
    tiny_vctk: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "reports"
    report, outputs = vctk_pool.prepare_speaker_pool(tiny_vctk, output_dir, top_k=2)
    with outputs[1].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, dialect="excel-tab"))
    assert len(rows) == report["summary"]["reported_candidates"] == 2
    assert list(rows[0]) == [
        "source_speaker_id",
        "rank",
        "age",
        "gender",
        "accent",
        "region",
        "source_utterance_id",
        "duration_sec",
        "transcript",
        "audio_path",
    ]


def test_no_source_audio_is_copied(tiny_vctk: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "reports"
    _, outputs = vctk_pool.prepare_speaker_pool(tiny_vctk, output_dir)
    assert {path.name for path in output_dir.iterdir()} == {
        "README.md",
        "candidate_report.json",
        "candidate_report.tsv",
    }
    assert not any(path.suffix in {".wav", ".flac"} for path in output_dir.rglob("*"))
    assert all(path.is_file() for path in outputs)


def test_output_paths_are_reproducible_and_not_absolute(tmp_path: Path) -> None:
    first_root = _build_tiny_vctk(tmp_path / "first" / "tiny-vctk")
    second_root = _build_tiny_vctk(tmp_path / "second" / "tiny-vctk")
    first = _report(first_root)
    second = _report(second_root)
    assert first == second
    assert not Path(first["source_dataset"]["source_root"]).is_absolute()
    for speaker in first["speakers"]:
        for candidate in speaker["candidates"]:
            assert not Path(candidate["audio_path"]).is_absolute()
            assert not Path(candidate["transcript_path"]).is_absolute()
