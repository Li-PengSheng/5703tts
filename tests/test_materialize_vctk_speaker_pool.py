"""Deterministic offline tests for VCTK speaker-pool materialisation.

The real corpus is never touched: every test builds a tiny synthetic VCTK tree,
candidate report, and selection manifest in ``tmp_path``. No TTS model is loaded.
"""

import array
import hashlib
import importlib.util
import json
import math
import wave
from pathlib import Path
from typing import Any

import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "materialize_vctk_speaker_pool.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("vctk_materialize", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
materialize = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(materialize)

SOURCE_SAMPLE_RATE = 48_000
SPEAKER_POOL_VERSION = "vctk_v0.1"
# Transcripts deliberately contain punctuation and an apostrophe so byte-exact
# preservation is observable.
SOURCE_SPEAKERS: tuple[
    tuple[str, dict[str, str], tuple[tuple[str, float, str], ...]], ...
] = (
    (
        "p101",
        {"age": "23", "gender": "F", "accent": "Welsh", "region": "Cardiff"},
        (
            ("p101_001", 4.0, "The occasion was the conference, wasn't it?"),
            ("p101_002", 5.0, "However, the decision has been welcomed here."),
            ("p101_003", 6.0, "Nobody expected the third recording to be chosen."),
        ),
    ),
    (
        "p102",
        {"age": "26", "gender": "M", "accent": "Indian"},
        (
            ("p102_001", 3.5, "She put a brave face on the departure yesterday."),
            ("p102_002", 4.5, "They said the report would arrive before Friday."),
        ),
    ),
    (
        "p103",
        {"age": "22", "gender": "M", "accent": "Scottish", "region": "Fife"},
        (
            ("p103_001", 5.5, "He asked whether the meeting could start later."),
            ("p103_002", 4.25, "We agreed to review the recordings together."),
        ),
    ),
)
DEFAULT_SELECTION = (
    {
        "speaker_id": "spk_001",
        "source_speaker_id": "p101",
        "primary_utterance_id": "p101_001",
        "backup_utterance_id": "p101_002",
    },
    {
        "speaker_id": "spk_002",
        "source_speaker_id": "p102",
        "primary_utterance_id": "p102_001",
        "backup_utterance_id": "p102_002",
    },
)


def _write_source_wav(
    path: Path, duration_sec: float, *, frequency: float, sample_rate: int
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = round(duration_sec * sample_rate)
    samples = array.array(
        "h",
        (
            int(12_000 * math.sin(2 * math.pi * frequency * index / sample_rate))
            for index in range(frame_count)
        ),
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return path


def _write_selection(
    path: Path,
    speakers: tuple[dict[str, str], ...] = DEFAULT_SELECTION,
    *,
    speaker_pool_version: str = SPEAKER_POOL_VERSION,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "speaker_pool_version": speaker_pool_version,
                "selection_status": "provisional_smoke_test_shortlist",
                "selection_method": "Manual speaker shortlist for smoke testing.",
                "speakers": list(speakers),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class Fixture:
    """Paths of one synthetic speaker-pool workspace."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.vctk_root = root / "tiny-vctk"
        self.output_dir = root / "pool"
        self.candidate_report = self.output_dir / "candidate_report.json"
        self.selection = self.output_dir / "selected_speakers.json"

    def materialise(self, path_root: Path | None = None) -> tuple[dict[str, Any], Path]:
        return materialize.materialise_speaker_pool(
            self.vctk_root,
            self.candidate_report,
            self.selection,
            self.output_dir,
            path_root=path_root or self.root,
        )

    def report(self) -> dict[str, Any]:
        return json.loads(self.candidate_report.read_text(encoding="utf-8"))

    def write_report(self, report: dict[str, Any]) -> None:
        self.candidate_report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def source_hashes(self) -> dict[str, tuple[int, str]]:
        return {
            path.relative_to(self.vctk_root).as_posix(): (
                path.stat().st_size,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in sorted(self.vctk_root.rglob("*"))
            if path.is_file()
        }


def _build_fixture(root: Path) -> Fixture:
    fixture = Fixture(root)
    speakers = []
    for index, (speaker_id, metadata, utterances) in enumerate(SOURCE_SPEAKERS):
        candidates = []
        for rank, (utterance_id, duration_sec, transcript) in enumerate(
            utterances, start=1
        ):
            audio_rel = f"wav48_silence_trimmed/{speaker_id}/{utterance_id}_mic1.wav"
            transcript_rel = f"txt/{speaker_id}/{utterance_id}.txt"
            _write_source_wav(
                fixture.vctk_root / audio_rel,
                duration_sec,
                frequency=110.0 + 40.0 * index + 7.0 * rank,
                sample_rate=SOURCE_SAMPLE_RATE,
            )
            transcript_path = fixture.vctk_root / transcript_rel
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            transcript_path.write_text(transcript + "\n", encoding="utf-8")
            candidates.append(
                {
                    "rank": rank,
                    "source_utterance_id": utterance_id,
                    "audio_path": audio_rel,
                    "transcript_path": transcript_rel,
                    "transcript": transcript,
                    "duration_sec": round(duration_sec, 6),
                    "sample_rate": SOURCE_SAMPLE_RATE,
                    "channels": 1,
                    "word_count": len(transcript.split()),
                }
            )
        speakers.append(
            {
                "source_speaker_id": speaker_id,
                "metadata": dict(metadata),
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
        )

    fixture.output_dir.mkdir(parents=True, exist_ok=True)
    fixture.write_report(
        {
            "speaker_pool_version": SPEAKER_POOL_VERSION,
            "source_dataset": {
                "name": "CSTR VCTK Corpus",
                "version": "0.92",
                "source_root": "tiny-vctk",
                "license": "CC BY 4.0",
            },
            "speakers": speakers,
        }
    )
    _write_selection(fixture.selection)
    return fixture


@pytest.fixture
def fixture(tmp_path: Path) -> Fixture:
    return _build_fixture(tmp_path / "workspace")


def _speaker(registry: dict[str, Any], speaker_id: str = "spk_001") -> dict[str, Any]:
    return next(
        entry for entry in registry["speakers"] if entry["speaker_id"] == speaker_id
    )


def test_selected_speaker_resolves_through_candidate_report(fixture: Fixture) -> None:
    registry, registry_path = fixture.materialise()

    assert registry_path == fixture.output_dir / "speaker_registry.json"
    assert json.loads(registry_path.read_text(encoding="utf-8")) == registry
    assert [entry["speaker_id"] for entry in registry["speakers"]] == [
        "spk_001",
        "spk_002",
    ]
    primary = _speaker(registry)["primary_reference"]
    assert primary["source_utterance_id"] == "p101_001"
    assert primary["candidate_rank"] == 1
    assert primary["source_audio_path"] == (
        "wav48_silence_trimmed/p101/p101_001_mic1.wav"
    )
    assert primary["source_transcript_path"] == "txt/p101/p101_001.txt"


def test_every_selected_speaker_is_materialised_without_skipping(
    fixture: Fixture,
) -> None:
    registry, _ = fixture.materialise()

    references = sorted(
        path.name for path in (fixture.output_dir / "references").iterdir()
    )
    assert registry["speaker_count"] == 2
    assert registry["reference_count"] == 4
    assert references == [
        "spk_001_backup.wav",
        "spk_001_primary.wav",
        "spk_002_backup.wav",
        "spk_002_primary.wav",
    ]


def test_missing_source_speaker_is_rejected(fixture: Fixture) -> None:
    _write_selection(
        fixture.selection,
        (
            {
                "speaker_id": "spk_001",
                "source_speaker_id": "p999",
                "primary_utterance_id": "p999_001",
                "backup_utterance_id": "p999_002",
            },
        ),
    )

    with pytest.raises(materialize.SpeakerPoolMaterialisationError) as error:
        fixture.materialise()

    assert "p999" in str(error.value)
    assert not (fixture.output_dir / "speaker_registry.json").exists()


def test_missing_utterance_is_rejected(fixture: Fixture) -> None:
    _write_selection(
        fixture.selection,
        (
            {
                "speaker_id": "spk_001",
                "source_speaker_id": "p101",
                "primary_utterance_id": "p101_404",
                "backup_utterance_id": "p101_002",
            },
        ),
    )

    with pytest.raises(materialize.SpeakerPoolMaterialisationError) as error:
        fixture.materialise()

    message = str(error.value)
    assert "p101_404" in message
    assert "p101_001" in message, "available candidates should be reported"


def test_duplicate_internal_speaker_id_is_rejected(fixture: Fixture) -> None:
    _write_selection(
        fixture.selection,
        (
            DEFAULT_SELECTION[0],
            {**DEFAULT_SELECTION[1], "speaker_id": "spk_001"},
        ),
    )

    with pytest.raises(
        materialize.SpeakerPoolMaterialisationError, match="Duplicate internal"
    ):
        fixture.materialise()


def test_duplicate_source_speaker_selection_is_rejected(fixture: Fixture) -> None:
    _write_selection(
        fixture.selection,
        (
            DEFAULT_SELECTION[0],
            {
                "speaker_id": "spk_002",
                "source_speaker_id": "p101",
                "primary_utterance_id": "p101_002",
                "backup_utterance_id": "p101_003",
            },
        ),
    )

    with pytest.raises(
        materialize.SpeakerPoolMaterialisationError, match="selected more than once"
    ):
        fixture.materialise()


def test_primary_and_backup_must_be_distinct(fixture: Fixture) -> None:
    _write_selection(
        fixture.selection,
        ({**DEFAULT_SELECTION[0], "backup_utterance_id": "p101_001"},),
    )

    with pytest.raises(
        materialize.SpeakerPoolMaterialisationError, match="must differ"
    ):
        fixture.materialise()


def test_speaker_pool_version_mismatch_is_rejected(fixture: Fixture) -> None:
    _write_selection(fixture.selection, speaker_pool_version="vctk_v0.9")

    with pytest.raises(materialize.SpeakerPoolMaterialisationError, match="vctk_v0.9"):
        fixture.materialise()


def test_exact_transcript_is_preserved(fixture: Fixture) -> None:
    registry, _ = fixture.materialise()

    for entry in registry["speakers"]:
        for role in ("primary", "backup"):
            reference = entry[f"{role}_reference"]
            source_text = (
                fixture.vctk_root / reference["source_transcript_path"]
            ).read_text(encoding="utf-8")
            assert reference["prompt_text"] == source_text.strip()
    assert _speaker(registry)["primary_reference"]["prompt_text"] == (
        "The occasion was the conference, wasn't it?"
    )


def test_stale_transcript_is_rejected(fixture: Fixture) -> None:
    (fixture.vctk_root / "txt/p101/p101_001.txt").write_text(
        "Someone edited this transcript after indexing.\n", encoding="utf-8"
    )

    with pytest.raises(
        materialize.SpeakerPoolMaterialisationError, match="no longer matches"
    ):
        fixture.materialise()


def test_missing_source_audio_is_rejected(fixture: Fixture) -> None:
    (fixture.vctk_root / "wav48_silence_trimmed/p101/p101_001_mic1.wav").unlink()

    with pytest.raises(
        materialize.SpeakerPoolMaterialisationError, match="source audio is missing"
    ):
        fixture.materialise()


def test_missing_source_transcript_is_rejected(fixture: Fixture) -> None:
    (fixture.vctk_root / "txt/p101/p101_001.txt").unlink()

    with pytest.raises(
        materialize.SpeakerPoolMaterialisationError,
        match="source transcript is missing",
    ):
        fixture.materialise()


@pytest.mark.parametrize("role", ["primary", "backup"])
def test_reference_audio_is_16_kilohertz_mono_pcm16(
    fixture: Fixture, role: str
) -> None:
    registry, _ = fixture.materialise()

    for entry in registry["speakers"]:
        reference = entry[f"{role}_reference"]
        path = fixture.root / reference["prompt_wav"]
        info = sf.info(path)
        assert info.samplerate == 16_000
        assert info.channels == 1
        assert info.subtype == "PCM_16"
        assert reference["materialised_sample_rate"] == 16_000
        assert reference["materialised_channels"] == 1
        assert reference["materialised_encoding"] == "PCM_16"
        assert path.stat().st_size > 0


def test_reference_duration_is_preserved_within_tolerance(fixture: Fixture) -> None:
    registry, _ = fixture.materialise()

    for entry in registry["speakers"]:
        for role in ("primary", "backup"):
            reference = entry[f"{role}_reference"]
            info = sf.info(fixture.root / reference["prompt_wav"])
            measured = info.frames / info.samplerate
            assert reference["original_sample_rate"] == SOURCE_SAMPLE_RATE
            assert abs(measured - reference["original_duration_sec"]) <= (
                materialize.DURATION_TOLERANCE_SEC
            )
            assert reference["materialised_duration_sec"] == pytest.approx(
                measured, abs=1e-6
            )


def test_registry_paths_are_relative_and_reproducible(tmp_path: Path) -> None:
    first = _build_fixture(tmp_path / "first")
    second = _build_fixture(tmp_path / "second")

    first_registry, _ = first.materialise()
    second_registry, _ = second.materialise()

    assert first_registry == second_registry
    assert first_registry["selection"]["manifest"] == "pool/selected_speakers.json"
    for entry in first_registry["speakers"]:
        for role in ("primary", "backup"):
            reference = entry[f"{role}_reference"]
            for field in (
                "prompt_wav",
                "source_audio_path",
                "source_transcript_path",
            ):
                assert not Path(reference[field]).is_absolute()
    assert str(tmp_path) not in json.dumps(first_registry)


def test_materialiser_follows_manual_primary_backup_not_candidate_rank(
    fixture: Fixture,
) -> None:
    _write_selection(
        fixture.selection,
        (
            {
                "speaker_id": "spk_001",
                "source_speaker_id": "p101",
                "primary_utterance_id": "p101_001",
                "backup_utterance_id": "p101_003",
            },
            {
                "speaker_id": "spk_002",
                "source_speaker_id": "p102",
                "primary_utterance_id": "p102_002",
                "backup_utterance_id": "p102_001",
            },
        ),
    )

    registry, _ = fixture.materialise()

    first = _speaker(registry, "spk_001")
    assert first["primary_reference"]["source_utterance_id"] == "p101_001"
    assert first["backup_reference"]["source_utterance_id"] == "p101_003"
    assert first["backup_reference"]["candidate_rank"] == 3
    second = _speaker(registry, "spk_002")
    assert second["primary_reference"]["source_utterance_id"] == "p102_002"
    assert second["backup_reference"]["source_utterance_id"] == "p102_001"
    assert second["primary_reference"]["candidate_rank"] == 2
    assert second["backup_reference"]["candidate_rank"] == 1
    references = fixture.output_dir / "references"
    assert (references / "spk_001_backup.wav").is_file()
    assert (references / "spk_002_primary.wav").is_file()


def test_registry_preserves_source_provenance(fixture: Fixture) -> None:
    registry, _ = fixture.materialise()

    entry = _speaker(registry)
    assert entry["speaker_id"] == "spk_001"
    assert entry["source_speaker_id"] == "p101"
    reference = entry["primary_reference"]
    assert reference["source_utterance_id"] == "p101_001"
    assert (fixture.vctk_root / reference["source_audio_path"]).is_file()
    assert (fixture.vctk_root / reference["source_transcript_path"]).is_file()
    assert (fixture.root / reference["prompt_wav"]).is_file()
    assert registry["source_dataset"]["doi"] == "10.7488/ds/2645"
    assert registry["source_dataset"]["license"] == "CC BY 4.0"
    assert registry["status"] == "provisional_smoke_test"


def test_registry_preserves_vctk_metadata_without_inference(fixture: Fixture) -> None:
    registry, _ = fixture.materialise()
    report_metadata = {
        speaker["source_speaker_id"]: speaker["metadata"]
        for speaker in fixture.report()["speakers"]
    }

    for entry in registry["speakers"]:
        metadata = entry["source_metadata"]
        assert metadata == report_metadata[entry["source_speaker_id"]]
        assert set(metadata) <= set(materialize.METADATA_FIELDS)
    # p102 has no region in the source metadata, so none may be invented.
    assert "region" not in _speaker(registry, "spk_002")["source_metadata"]


def test_materialised_sha256_is_recorded(fixture: Fixture) -> None:
    registry, _ = fixture.materialise()

    for entry in registry["speakers"]:
        for role in ("primary", "backup"):
            reference = entry[f"{role}_reference"]
            path = fixture.root / reference["prompt_wav"]
            assert reference["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            assert reference["materialised_bytes"] == path.stat().st_size


def test_source_audio_is_never_modified(fixture: Fixture) -> None:
    before = fixture.source_hashes()

    registry, _ = fixture.materialise()

    # Covers content changes and any file added to or removed from the corpus.
    assert fixture.source_hashes() == before
    for entry in registry["speakers"]:
        prompt_wav = (fixture.root / entry["primary_reference"]["prompt_wav"]).resolve()
        assert fixture.vctk_root.resolve() not in prompt_wav.parents


def test_ffmpeg_requirement_is_reported_clearly(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = fixture.report()
    for speaker in report["speakers"]:
        for candidate in speaker["candidates"]:
            source = fixture.vctk_root / candidate["audio_path"]
            flac_relative = candidate["audio_path"].replace(".wav", ".flac")
            (fixture.vctk_root / flac_relative).write_bytes(source.read_bytes())
            candidate["audio_path"] = flac_relative
    fixture.write_report(report)
    monkeypatch.setattr(materialize, "which", lambda binary: None)

    with pytest.raises(
        materialize.SpeakerPoolMaterialisationError, match="requires ffmpeg"
    ) as error:
        fixture.materialise()

    assert ".flac" in str(error.value)


def test_reference_failing_verification_fails_the_command(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def write_wrong_format(source_path: Path, destination: Path) -> None:
        _write_source_wav(destination, 1.0, frequency=200.0, sample_rate=8_000)

    monkeypatch.setattr(materialize, "_convert_to_reference", write_wrong_format)

    with pytest.raises(
        materialize.SpeakerPoolMaterialisationError, match="failed validation"
    ) as error:
        fixture.materialise()

    assert "sample rate is 8000" in str(error.value)


def test_cli_reports_failure_with_a_nonzero_exit_code(
    fixture: Fixture, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_selection(fixture.selection, speaker_pool_version="vctk_v9.9")

    exit_code = materialize.main(
        [
            "--vctk-root",
            str(fixture.vctk_root),
            "--candidate-report",
            str(fixture.candidate_report),
            "--selection",
            str(fixture.selection),
            "--output-dir",
            str(fixture.output_dir),
        ]
    )

    assert exit_code == 2
    assert "materialisation failed" in capsys.readouterr().err
