"""Materialise CosyVoice reference audio for the selected VCTK speaker pool.

This stage resolves a manual speaker shortlist through the existing candidate
report, converts the referenced VCTK recordings into one consistent prompt
format, and records the provenance needed to trace every reference back to its
original VCTK file and transcript. Nothing here is integrated into the dialogue
pipeline, and no speaker is judged or reranked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import soundfile as sf
from pydub import AudioSegment
from pydub.utils import which

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEAKER_POOL_VERSION = "vctk_v0.1"
REGISTRY_STATUS = "provisional_smoke_test"
DEFAULT_VCTK_ROOT = Path("vctk-speaker")
DEFAULT_OUTPUT_DIR = Path("data/speaker_pool/vctk_v0.1")
DEFAULT_CANDIDATE_REPORT = DEFAULT_OUTPUT_DIR / "candidate_report.json"
DEFAULT_SELECTION = DEFAULT_OUTPUT_DIR / "selected_speakers.json"
REGISTRY_FILENAME = "speaker_registry.json"
REFERENCES_DIRNAME = "references"
REFERENCE_ROLES = ("primary", "backup")
REFERENCE_SAMPLE_RATE = 16_000
REFERENCE_CHANNELS = 1
REFERENCE_SAMPLE_WIDTH_BYTES = 2
REFERENCE_SUBTYPE = "PCM_16"
DURATION_TOLERANCE_SEC = 0.05
METADATA_FIELDS = ("age", "gender", "accent", "region")
NATIVE_DECODE_SUFFIXES = {".wav"}
FFMPEG_BINARIES = ("ffmpeg", "avconv")
SOURCE_DATASET_NAME = "CSTR VCTK Corpus"
SOURCE_DATASET_VERSION = "0.92"
SOURCE_LICENSE = "CC BY 4.0"
SOURCE_DOI = "10.7488/ds/2645"
SOURCE_CITATION = (
    "Yamagishi, Junichi; Veaux, Christophe; MacDonald, Kirsten. (2019). "
    "CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning "
    "Toolkit (version 0.92). University of Edinburgh, CSTR. "
    "DOI: 10.7488/ds/2645"
)
CONVERSION_DESCRIPTION = (
    "Channel, sample-rate, and bit-depth conversion only. No denoising, pitch "
    "shifting, speed change, loudness enhancement, telephone filtering, added "
    "silence, or speech-content change is applied."
)


class SpeakerPoolMaterialisationError(ValueError):
    """Raised when a selected reference cannot be materialised safely."""


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SpeakerPoolMaterialisationError(f"{label} not found: {path}") from error
    except json.JSONDecodeError as error:
        raise SpeakerPoolMaterialisationError(
            f"{label} is not valid JSON ({path}): {error}"
        ) from error
    if not isinstance(value, dict):
        raise SpeakerPoolMaterialisationError(
            f"{label} must contain a JSON object: {path}"
        )
    return value


def load_candidate_report(path: Path) -> dict[str, Any]:
    """Load the candidate report that remains the source of truth for provenance."""
    report = _load_json_object(path, "Candidate report")
    speakers = report.get("speakers")
    if not isinstance(speakers, list) or not speakers:
        raise SpeakerPoolMaterialisationError(
            f"Candidate report has no indexed speakers: {path}"
        )
    return report


def load_selection(path: Path) -> dict[str, Any]:
    """Load the frozen manual shortlist without reranking or replacing it."""
    selection = _load_json_object(path, "Selection manifest")
    speakers = selection.get("speakers")
    if not isinstance(speakers, list) or not speakers:
        raise SpeakerPoolMaterialisationError(
            f"Selection manifest lists no speakers: {path}"
        )
    return selection


def _required_field(entry: dict[str, Any], field: str, position: int) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SpeakerPoolMaterialisationError(
            f"Selection entry {position} must provide a non-empty {field}"
        )
    return value


def _candidates_by_utterance(speaker: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        candidate["source_utterance_id"]: candidate
        for candidate in speaker.get("candidates", [])
    }


def resolve_selection(
    report: dict[str, Any], selection: dict[str, Any]
) -> list[dict[str, Any]]:
    """Resolve every selected speaker and utterance through the candidate report.

    Rejecting the whole selection on the first unresolvable entry is deliberate:
    a shortlist that no longer matches the indexed corpus must be reviewed by a
    human rather than silently materialised with a missing speaker.
    """
    report_version = report.get("speaker_pool_version")
    selection_version = selection.get("speaker_pool_version")
    if selection_version != report_version:
        raise SpeakerPoolMaterialisationError(
            f"Selection manifest targets speaker pool {selection_version!r} but the "
            f"candidate report describes {report_version!r}"
        )

    speakers_by_id = {
        speaker["source_speaker_id"]: speaker for speaker in report["speakers"]
    }
    resolved: list[dict[str, Any]] = []
    seen_speaker_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    for position, entry in enumerate(selection["speakers"], start=1):
        if not isinstance(entry, dict):
            raise SpeakerPoolMaterialisationError(
                f"Selection entry {position} must be a JSON object"
            )
        speaker_id = _required_field(entry, "speaker_id", position)
        source_speaker_id = _required_field(entry, "source_speaker_id", position)
        utterance_ids = {
            role: _required_field(entry, f"{role}_utterance_id", position)
            for role in REFERENCE_ROLES
        }

        if speaker_id in seen_speaker_ids:
            raise SpeakerPoolMaterialisationError(
                f"Duplicate internal speaker_id in selection: {speaker_id}"
            )
        seen_speaker_ids.add(speaker_id)
        if source_speaker_id in seen_source_ids:
            raise SpeakerPoolMaterialisationError(
                f"Source speaker {source_speaker_id} is selected more than once; "
                "each VCTK speaker may back only one internal speaker_id"
            )
        seen_source_ids.add(source_speaker_id)
        if utterance_ids["primary"] == utterance_ids["backup"]:
            raise SpeakerPoolMaterialisationError(
                f"{speaker_id}: primary and backup utterances must differ; both are "
                f"{utterance_ids['primary']}"
            )

        speaker = speakers_by_id.get(source_speaker_id)
        if speaker is None:
            raise SpeakerPoolMaterialisationError(
                f"{speaker_id}: source speaker {source_speaker_id} is not in the "
                "candidate report; re-run scripts/prepare_vctk_speaker_pool.py or "
                "correct the selection manifest"
            )
        candidates = _candidates_by_utterance(speaker)
        references: dict[str, dict[str, Any]] = {}
        for role, utterance_id in utterance_ids.items():
            candidate = candidates.get(utterance_id)
            if candidate is None:
                raise SpeakerPoolMaterialisationError(
                    f"{speaker_id}: {role} utterance {utterance_id} is not a reported "
                    f"candidate for {source_speaker_id}; available candidates: "
                    f"{sorted(candidates)}"
                )
            references[role] = candidate
        resolved.append(
            {
                "speaker_id": speaker_id,
                "source_speaker_id": source_speaker_id,
                "source_metadata": {
                    field: speaker.get("metadata", {})[field]
                    for field in METADATA_FIELDS
                    if field in speaker.get("metadata", {})
                },
                "references": references,
            }
        )
    return resolved


def _resolve_source_paths(
    vctk_root: Path, speaker_id: str, role: str, candidate: dict[str, Any]
) -> tuple[Path, Path]:
    audio_path = vctk_root / candidate["audio_path"]
    transcript_path = vctk_root / candidate["transcript_path"]
    if not audio_path.is_file():
        raise SpeakerPoolMaterialisationError(
            f"{speaker_id} {role}: source audio is missing: {audio_path}. Pass "
            "--vctk-root pointing to the extracted VCTK Corpus 0.92 root."
        )
    if not transcript_path.is_file():
        raise SpeakerPoolMaterialisationError(
            f"{speaker_id} {role}: source transcript is missing: {transcript_path}"
        )
    return audio_path, transcript_path


def read_transcript(
    transcript_path: Path, candidate: dict[str, Any], speaker_id: str, role: str
) -> str:
    """Read the transcript from VCTK itself so no prompt text is hard-coded here."""
    try:
        transcript = transcript_path.read_text(encoding="utf-8-sig").strip()
    except (OSError, UnicodeError) as error:
        raise SpeakerPoolMaterialisationError(
            f"{speaker_id} {role}: unable to read transcript {transcript_path}: {error}"
        ) from error
    if not transcript:
        raise SpeakerPoolMaterialisationError(
            f"{speaker_id} {role}: transcript is empty: {transcript_path}"
        )
    reported = candidate.get("transcript")
    if reported is not None and transcript != reported:
        raise SpeakerPoolMaterialisationError(
            f"{speaker_id} {role}: transcript at {transcript_path} no longer matches "
            "the candidate report; re-run scripts/prepare_vctk_speaker_pool.py"
        )
    return transcript


def probe_source_audio(
    audio_path: Path, candidate: dict[str, Any], speaker_id: str, role: str
) -> tuple[float, int, int]:
    """Measure the source recording instead of trusting the report alone."""
    try:
        info = sf.info(audio_path)
    except (OSError, RuntimeError, sf.LibsndfileError) as error:
        raise SpeakerPoolMaterialisationError(
            f"{speaker_id} {role}: unable to read source audio {audio_path}: {error}"
        ) from error
    duration_sec = float(info.duration)
    reported = candidate.get("duration_sec")
    if reported is not None and abs(duration_sec - float(reported)) > (
        DURATION_TOLERANCE_SEC
    ):
        raise SpeakerPoolMaterialisationError(
            f"{speaker_id} {role}: source audio {audio_path} is "
            f"{duration_sec:.6f}s but the candidate report recorded "
            f"{float(reported):.6f}s; re-run scripts/prepare_vctk_speaker_pool.py"
        )
    return duration_sec, int(info.samplerate), int(info.channels)


def _ffmpeg_binary() -> str | None:
    for binary in FFMPEG_BINARIES:
        if which(binary):
            return binary
    return None


def require_decoder(source_paths: list[Path]) -> tuple[bool, str]:
    """Fail early when the selected source formats need an unavailable decoder."""
    suffixes = sorted(
        {
            path.suffix.lower()
            for path in source_paths
            if path.suffix.lower() not in NATIVE_DECODE_SUFFIXES
        }
    )
    if not suffixes:
        return False, "builtin_wav"
    if _ffmpeg_binary() is None:
        raise SpeakerPoolMaterialisationError(
            f"Decoding selected source audio ({', '.join(suffixes)}) requires ffmpeg, "
            "which was not found on PATH. Install ffmpeg (for example "
            "'sudo apt install ffmpeg') and re-run this command."
        )
    return True, "ffmpeg"


def _convert_to_reference(source_path: Path, destination: Path) -> None:
    """Convert one recording to the shared prompt format and nothing more."""
    try:
        segment = AudioSegment.from_file(source_path)
        segment = (
            segment.set_channels(REFERENCE_CHANNELS)
            .set_sample_width(REFERENCE_SAMPLE_WIDTH_BYTES)
            .set_frame_rate(REFERENCE_SAMPLE_RATE)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        segment.export(destination, format="wav")
    # Decoder and encoder failures arrive as several unrelated exception types;
    # all of them are only useful with the failing source file attached.
    except Exception as error:
        raise SpeakerPoolMaterialisationError(
            f"Unable to convert {source_path} to {destination}: "
            f"{type(error).__name__}: {error}"
        ) from error


def verify_reference(
    destination: Path, source_duration_sec: float, speaker_id: str, role: str
) -> dict[str, Any]:
    """Validate one written reference; a bad reference must fail the command."""
    label = f"{speaker_id} {role} reference {destination}"
    if not destination.is_file():
        raise SpeakerPoolMaterialisationError(f"{label} was not written")
    size_bytes = destination.stat().st_size
    if size_bytes == 0:
        raise SpeakerPoolMaterialisationError(f"{label} is empty")
    try:
        info = sf.info(destination)
    except (OSError, RuntimeError, sf.LibsndfileError) as error:
        raise SpeakerPoolMaterialisationError(
            f"{label} is not readable audio: {error}"
        ) from error

    duration_sec = info.frames / info.samplerate if info.samplerate else 0.0
    problems = []
    if info.samplerate != REFERENCE_SAMPLE_RATE:
        problems.append(
            f"sample rate is {info.samplerate}, expected {REFERENCE_SAMPLE_RATE}"
        )
    if info.channels != REFERENCE_CHANNELS:
        problems.append(f"channel count is {info.channels}, expected mono")
    # Some containers expose no subtype; only an inspectable subtype is checked.
    if info.subtype and info.subtype != REFERENCE_SUBTYPE:
        problems.append(f"subtype is {info.subtype}, expected {REFERENCE_SUBTYPE}")
    if duration_sec <= 0:
        problems.append("duration is not positive")
    elif abs(duration_sec - source_duration_sec) > DURATION_TOLERANCE_SEC:
        problems.append(
            f"duration is {duration_sec:.6f}s but the source is "
            f"{source_duration_sec:.6f}s, beyond the {DURATION_TOLERANCE_SEC}s "
            "resampling tolerance"
        )
    if problems:
        raise SpeakerPoolMaterialisationError(
            f"{label} failed validation: {'; '.join(problems)}"
        )
    return {
        "duration_sec": round(duration_sec, 6),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "encoding": info.subtype or REFERENCE_SUBTYPE,
        "bytes": size_bytes,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to(path: Path, root: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise SpeakerPoolMaterialisationError(
            f"{label} must live inside {root} so registry paths stay reproducible: "
            f"{path}"
        ) from error


def _portable_source_root(vctk_root: Path) -> str:
    resolved = vctk_root.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.name


def materialise_speaker_pool(
    vctk_root: Path,
    candidate_report_path: Path,
    selection_path: Path,
    output_dir: Path,
    *,
    path_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], Path]:
    """Materialise every selected reference and write the speaker registry."""
    report = load_candidate_report(candidate_report_path)
    selection = load_selection(selection_path)
    resolved = resolve_selection(report, selection)

    sources: dict[tuple[str, str], dict[str, Any]] = {}
    for speaker in resolved:
        for role, candidate in speaker["references"].items():
            audio_path, transcript_path = _resolve_source_paths(
                vctk_root, speaker["speaker_id"], role, candidate
            )
            sources[speaker["speaker_id"], role] = {
                "candidate": candidate,
                "audio_path": audio_path,
                "transcript_path": transcript_path,
            }
    decoder_required, decoder = require_decoder(
        [source["audio_path"] for source in sources.values()]
    )

    references_dir = output_dir / REFERENCES_DIRNAME
    references_dir.mkdir(parents=True, exist_ok=True)
    registry_speakers: list[dict[str, Any]] = []
    for speaker in resolved:
        speaker_id = speaker["speaker_id"]
        entry: dict[str, Any] = {
            "speaker_id": speaker_id,
            "source_speaker_id": speaker["source_speaker_id"],
            "source_metadata": speaker["source_metadata"],
        }
        for role in REFERENCE_ROLES:
            source = sources[speaker_id, role]
            candidate = source["candidate"]
            transcript = read_transcript(
                source["transcript_path"], candidate, speaker_id, role
            )
            source_duration, source_sample_rate, source_channels = probe_source_audio(
                source["audio_path"], candidate, speaker_id, role
            )
            destination = references_dir / f"{speaker_id}_{role}.wav"
            _convert_to_reference(source["audio_path"], destination)
            measured = verify_reference(destination, source_duration, speaker_id, role)
            entry[f"{role}_reference"] = {
                "source_utterance_id": candidate["source_utterance_id"],
                "candidate_rank": candidate.get("rank"),
                "source_audio_path": candidate["audio_path"],
                "source_transcript_path": candidate["transcript_path"],
                "prompt_wav": _relative_to(
                    destination, path_root, "Reference audio directory"
                ),
                "prompt_text": transcript,
                "original_duration_sec": round(source_duration, 6),
                "original_sample_rate": source_sample_rate,
                "original_channels": source_channels,
                "materialised_duration_sec": measured["duration_sec"],
                "materialised_sample_rate": measured["sample_rate"],
                "materialised_channels": measured["channels"],
                "materialised_encoding": measured["encoding"],
                "materialised_bytes": measured["bytes"],
                "sha256": _sha256(destination),
            }
        registry_speakers.append(entry)

    registry = {
        "speaker_pool_version": SPEAKER_POOL_VERSION,
        "status": REGISTRY_STATUS,
        "generated_by": "scripts/materialize_vctk_speaker_pool.py",
        "integration_status": (
            "Not integrated into the dialogue pipeline; no dialogue speaker "
            "assignment is implied by this registry."
        ),
        "source_dataset": {
            "name": SOURCE_DATASET_NAME,
            "version": SOURCE_DATASET_VERSION,
            "license": SOURCE_LICENSE,
            "doi": SOURCE_DOI,
            "citation": SOURCE_CITATION,
            "source_root": _portable_source_root(vctk_root),
        },
        "selection": {
            "manifest": _relative_to(selection_path, path_root, "Selection manifest"),
            "candidate_report": _relative_to(
                candidate_report_path, path_root, "Candidate report"
            ),
            "selection_status": selection.get("selection_status"),
            "selection_method": selection.get("selection_method"),
        },
        "reference_audio_format": {
            "sample_rate": REFERENCE_SAMPLE_RATE,
            "channels": REFERENCE_CHANNELS,
            "encoding": REFERENCE_SUBTYPE,
            "container": "wav",
            "conversion": CONVERSION_DESCRIPTION,
            "duration_tolerance_sec": DURATION_TOLERANCE_SEC,
        },
        "materialisation": {
            "tool": "pydub",
            "decoder": decoder,
            "ffmpeg_required": decoder_required,
        },
        "paths": {
            "source_paths_relative_to": _portable_source_root(vctk_root),
            "materialised_paths_relative_to": "project_root",
        },
        "metadata_policy": (
            "Demographic fields are copied verbatim from VCTK speaker-info.txt via "
            "candidate_report.json; nothing is inferred from audio."
        ),
        "speaker_count": len(registry_speakers),
        "reference_count": len(registry_speakers) * len(REFERENCE_ROLES),
        "speakers": registry_speakers,
    }
    registry_path = output_dir / REGISTRY_FILENAME
    registry_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return registry, registry_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialise CosyVoice reference audio for selected VCTK speakers"
    )
    parser.add_argument("--vctk-root", type=Path, default=DEFAULT_VCTK_ROOT)
    parser.add_argument(
        "--candidate-report", type=Path, default=DEFAULT_CANDIDATE_REPORT
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        registry, registry_path = materialise_speaker_pool(
            args.vctk_root,
            args.candidate_report,
            args.selection,
            args.output_dir,
        )
    except (SpeakerPoolMaterialisationError, OSError) as error:
        print(f"Speaker pool materialisation failed: {error}", file=sys.stderr)
        return 2

    fmt = registry["reference_audio_format"]
    print(f"Selected speakers: {registry['speaker_count']}")
    print(f"Reference files: {registry['reference_count']}")
    print(
        f"Reference format: {fmt['sample_rate']} Hz, "
        f"{fmt['channels']} channel, {fmt['encoding']}"
    )
    print(f"Decoder: {registry['materialisation']['decoder']}")
    print(f"Reference directory: {args.output_dir / REFERENCES_DIRNAME}")
    print(f"Wrote: {registry_path}")
    print("Reference audio is regenerated locally and is not committed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
