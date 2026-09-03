"""Index local VCTK 0.92 audio and produce reference-candidate reports."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import soundfile as sf

SPEAKER_POOL_VERSION = "vctk_v0.1"
SOURCE_DATASET_NAME = "CSTR VCTK Corpus"
SOURCE_DATASET_VERSION = "0.92"
SOURCE_LICENSE = "CC BY 4.0"
DEFAULT_VCTK_ROOT = Path("vctk-speaker")
DEFAULT_OUTPUT_DIR = Path("data/speaker_pool/vctk_v0.1")
DEFAULT_MIC = "mic1"
DEFAULT_MIN_DURATION = 3.0
DEFAULT_MAX_DURATION = 8.0
DEFAULT_TOP_K = 5
RANKING_DURATION_CENTER_SEC = 5.0
RANKING_WORD_COUNT_CENTER = 12
MIN_TRANSCRIPT_WORDS = 5
MAX_TRANSCRIPT_WORDS_PER_SEC = 4.5
AUDIO_EXTENSIONS = {".flac", ".wav"}
EXCLUDED_SPEAKERS = {
    "p280": "Excluded from v0.1 because of known microphone recording issues.",
    "p315": (
        "Excluded because transcript data is unavailable and microphone recording "
        "issues are documented."
    ),
}
WORD_PATTERN = re.compile(r"[\w]+(?:['’-][\w]+)*", re.UNICODE)
MIC_FILENAME_PATTERN = re.compile(
    r"^(?P<utterance>.+)_(?P<mic>mic[^.]+)(?P<extension>\.[^.]+)$",
    re.IGNORECASE,
)


class VCTKPreparationError(ValueError):
    """Raised when the local source cannot be indexed safely."""


def validate_source_structure(vctk_root: Path) -> dict[str, Path]:
    """Validate required VCTK inputs and return their locations."""
    required = {
        "speaker_info": vctk_root / "speaker-info.txt",
        "transcripts": vctk_root / "txt",
        "audio": vctk_root / "wav48_silence_trimmed",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise VCTKPreparationError(
            "VCTK source structure is incomplete. Missing required path(s):\n"
            f"  - {formatted}\n"
            "Pass --vctk-root pointing to the extracted VCTK Corpus 0.92 root."
        )
    if not required["speaker_info"].is_file():
        raise VCTKPreparationError(
            f"Expected a file at {required['speaker_info']}, but found another type."
        )
    for key in ("transcripts", "audio"):
        if not required[key].is_dir():
            raise VCTKPreparationError(
                f"Expected a directory at {required[key]}, but found another type."
            )
    return required


def _normalise_metadata_field(field: str) -> str:
    aliases = {"id": "source_speaker_id", "accents": "accent"}
    return aliases.get(field.strip().lower(), field.strip().lower())


def parse_speaker_info(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Parse VCTK's whitespace metadata while retaining only supplied values.

    VCTK 0.92 stores ID, age, gender, and accent as single tokens. The remaining
    text is the region, except for an optional parenthesized source comment.
    """
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    nonempty = [line.rstrip() for line in lines if line.strip()]
    if not nonempty:
        raise VCTKPreparationError(f"Speaker metadata file is empty: {path}")

    header_tokens = nonempty[0].split()
    header_fields = [_normalise_metadata_field(token) for token in header_tokens]
    required_fields = {"source_speaker_id", "age", "gender", "accent"}
    if not required_fields.issubset(header_fields):
        raise VCTKPreparationError(
            "speaker-info.txt must provide ID, AGE, GENDER, and ACCENTS columns; "
            f"found: {header_tokens}"
        )

    has_region = "region" in header_fields
    has_comments = "comments" in header_fields
    speakers: list[dict[str, str]] = []
    for line_number, line in enumerate(nonempty[1:], start=2):
        values = line.split(maxsplit=4)
        if len(values) < 4:
            raise VCTKPreparationError(
                f"Malformed speaker metadata at {path}:{line_number}: {line!r}"
            )
        speaker_id, age, gender, accent = values[:4]
        metadata = {
            "source_speaker_id": speaker_id,
            "age": age,
            "gender": gender,
            "accent": accent,
        }
        remainder = values[4].strip() if len(values) == 5 else ""
        comment = ""
        if has_comments and remainder:
            comment_match = re.search(r"\s*(\([^)]*\))\s*$", remainder)
            if comment_match:
                comment = comment_match.group(1)
                remainder = remainder[: comment_match.start()].strip()
        if has_region and remainder:
            metadata["region"] = remainder
        if has_comments and comment:
            metadata["comments"] = comment
        speakers.append(metadata)

    speaker_ids = [speaker["source_speaker_id"] for speaker in speakers]
    if len(speaker_ids) != len(set(speaker_ids)):
        raise VCTKPreparationError("speaker-info.txt contains duplicate speaker IDs.")
    return speakers, header_fields


def _portable_source_root(vctk_root: Path) -> str:
    resolved = vctk_root.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.name


def _word_count(transcript: str) -> int:
    return len(WORD_PATTERN.findall(transcript))


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        abs(candidate["duration_sec"] - RANKING_DURATION_CENTER_SEC),
        abs(candidate["word_count"] - RANKING_WORD_COUNT_CENTER),
        candidate["source_utterance_id"],
        candidate["audio_path"],
    )


def _requested_audio_files(
    speaker_audio_dir: Path, mic: str
) -> tuple[dict[str, Path], int, int]:
    selected: dict[str, Path] = {}
    other_mic_count = 0
    duplicate_count = 0
    if not speaker_audio_dir.is_dir():
        return selected, other_mic_count, duplicate_count

    for path in sorted(speaker_audio_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        match = MIC_FILENAME_PATTERN.match(path.name)
        if match is None:
            continue
        if match.group("mic").lower() != mic.lower():
            other_mic_count += 1
            continue
        utterance_id = match.group("utterance")
        if utterance_id in selected:
            duplicate_count += 1
            continue
        selected[utterance_id] = path
    return selected, other_mic_count, duplicate_count


def _relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _new_counts() -> dict[str, int]:
    return {
        "requested_mic_audio": 0,
        "other_mic_audio_skipped": 0,
        "duplicate_audio_skipped": 0,
        "missing_audio": 0,
        "missing_transcript": 0,
        "unreadable_transcript": 0,
        "matched_utterances": 0,
        "unreadable_audio": 0,
        "empty_transcript": 0,
        "very_short_transcript": 0,
        "transcript_too_long_for_duration": 0,
        "duration_outside_range": 0,
        "candidate_utterances": 0,
        "reported_candidates": 0,
    }


def _add_counts(total: dict[str, int], speaker_counts: dict[str, int]) -> None:
    for key, value in speaker_counts.items():
        total[key] += value


def _index_speaker(
    vctk_root: Path,
    speaker: dict[str, str],
    *,
    mic: str,
    min_duration: float,
    max_duration: float,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    speaker_id = speaker["source_speaker_id"]
    audio_dir = vctk_root / "wav48_silence_trimmed" / speaker_id
    transcript_dir = vctk_root / "txt" / speaker_id
    audio_by_id, other_mic_count, duplicate_count = _requested_audio_files(
        audio_dir, mic
    )
    counts = _new_counts()
    counts["requested_mic_audio"] = len(audio_by_id)
    counts["other_mic_audio_skipped"] = other_mic_count
    counts["duplicate_audio_skipped"] = duplicate_count

    transcript_ids = (
        {path.stem for path in transcript_dir.glob("*.txt") if path.is_file()}
        if transcript_dir.is_dir()
        else set()
    )
    counts["missing_audio"] = len(transcript_ids - set(audio_by_id))
    candidates: list[dict[str, Any]] = []
    for utterance_id, audio_path in sorted(audio_by_id.items()):
        transcript_path = transcript_dir / f"{utterance_id}.txt"
        if not transcript_path.is_file():
            counts["missing_transcript"] += 1
            continue
        try:
            transcript = transcript_path.read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeError):
            counts["unreadable_transcript"] += 1
            continue
        counts["matched_utterances"] += 1
        if not transcript:
            counts["empty_transcript"] += 1
            continue
        word_count = _word_count(transcript)
        if word_count < MIN_TRANSCRIPT_WORDS:
            counts["very_short_transcript"] += 1
            continue
        try:
            audio_info = sf.info(audio_path)
        except (OSError, RuntimeError, sf.LibsndfileError):
            counts["unreadable_audio"] += 1
            continue
        duration_sec = float(audio_info.duration)
        if not min_duration <= duration_sec <= max_duration:
            counts["duration_outside_range"] += 1
            continue
        if word_count / duration_sec > MAX_TRANSCRIPT_WORDS_PER_SEC:
            counts["transcript_too_long_for_duration"] += 1
            continue
        candidates.append(
            {
                "source_utterance_id": utterance_id,
                "audio_path": _relative_posix(audio_path, vctk_root),
                "transcript_path": _relative_posix(transcript_path, vctk_root),
                "transcript": transcript,
                "duration_sec": round(duration_sec, 6),
                "sample_rate": int(audio_info.samplerate),
                "channels": int(audio_info.channels),
                "word_count": word_count,
            }
        )

    candidates.sort(key=_candidate_sort_key)
    counts["candidate_utterances"] = len(candidates)
    reported = [
        {"rank": rank, **candidate}
        for rank, candidate in enumerate(candidates[:top_k], start=1)
    ]
    counts["reported_candidates"] = len(reported)
    metadata = {
        key: value for key, value in speaker.items() if key != "source_speaker_id"
    }
    return (
        {
            "source_speaker_id": speaker_id,
            "metadata": metadata,
            "candidate_count": len(candidates),
            "candidates": reported,
            "index_counts": counts,
        },
        counts,
    )


def build_candidate_report(
    vctk_root: Path,
    *,
    mic: str = DEFAULT_MIC,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[dict[str, Any], list[str]]:
    """Index a VCTK root and return a deterministic candidate report."""
    paths = validate_source_structure(vctk_root)
    if not mic.strip():
        raise VCTKPreparationError("--mic must be a non-empty microphone label.")
    if min_duration < 0 or max_duration < min_duration:
        raise VCTKPreparationError(
            "Duration bounds must satisfy 0 <= --min-duration <= --max-duration."
        )
    if top_k < 1:
        raise VCTKPreparationError("--top-k must be at least 1.")

    source_speakers, metadata_fields = parse_speaker_info(paths["speaker_info"])
    eligible_speakers = [
        speaker
        for speaker in source_speakers
        if speaker["source_speaker_id"] not in EXCLUDED_SPEAKERS
    ]
    totals = _new_counts()
    indexed_speakers: list[dict[str, Any]] = []
    for speaker in sorted(
        eligible_speakers, key=lambda item: item["source_speaker_id"]
    ):
        indexed, counts = _index_speaker(
            vctk_root,
            speaker,
            mic=mic,
            min_duration=min_duration,
            max_duration=max_duration,
            top_k=top_k,
        )
        indexed_speakers.append(indexed)
        _add_counts(totals, counts)

    report = {
        "speaker_pool_version": SPEAKER_POOL_VERSION,
        "source_dataset": {
            "name": SOURCE_DATASET_NAME,
            "version": SOURCE_DATASET_VERSION,
            "source_root": _portable_source_root(vctk_root),
            "license": SOURCE_LICENSE,
            "excluded_speakers": [
                {"source_speaker_id": speaker_id, "reason": reason}
                for speaker_id, reason in EXCLUDED_SPEAKERS.items()
            ],
        },
        "selection_config": {
            "microphone": mic,
            "min_duration_sec": min_duration,
            "max_duration_sec": max_duration,
            "min_transcript_words": MIN_TRANSCRIPT_WORDS,
            "max_transcript_words_per_sec": MAX_TRANSCRIPT_WORDS_PER_SEC,
            "ranking_duration_center_sec": RANKING_DURATION_CENTER_SEC,
            "ranking_word_count_center": RANKING_WORD_COUNT_CENTER,
            "ranking_method": (
                "Ascending distance from duration center, then word-count center, "
                "then utterance ID and audio path; this is heuristic ranking, not "
                "perceptual-quality scoring."
            ),
            "top_k_per_speaker": top_k,
        },
        "summary": {
            "speakers_indexed": len(source_speakers),
            "speakers_eligible": len(eligible_speakers),
            "speakers_with_candidates": sum(
                speaker["candidate_count"] > 0 for speaker in indexed_speakers
            ),
            "matched_utterances": totals["matched_utterances"],
            "candidate_utterances": totals["candidate_utterances"],
            "reported_candidates": totals["reported_candidates"],
            "index_counts": totals,
        },
        "speakers": indexed_speakers,
    }
    demographic_fields = [
        field
        for field in ("age", "gender", "accent", "region")
        if field in metadata_fields
    ]
    return report, demographic_fields


def _write_tsv(
    path: Path, report: dict[str, Any], demographic_fields: list[str]
) -> None:
    fields = [
        "source_speaker_id",
        "rank",
        *demographic_fields,
        "source_utterance_id",
        "duration_sec",
        "transcript",
        "audio_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, dialect="excel-tab")
        writer.writeheader()
        for speaker in report["speakers"]:
            for candidate in speaker["candidates"]:
                row = {
                    "source_speaker_id": speaker["source_speaker_id"],
                    "rank": candidate["rank"],
                    **{
                        field: speaker["metadata"].get(field, "")
                        for field in demographic_fields
                    },
                    "source_utterance_id": candidate["source_utterance_id"],
                    "duration_sec": candidate["duration_sec"],
                    "transcript": candidate["transcript"],
                    "audio_path": candidate["audio_path"],
                }
                writer.writerow(row)


def _readme_text(report: dict[str, Any]) -> str:
    config = report["selection_config"]
    return f"""# VCTK Speaker Pool v0.1

This directory holds every speaker-pool artefact for v0.1. The pool is
**provisional**: no speaker has been accepted yet, and nothing here is
integrated into the dialogue pipeline or used for dialogue speaker assignment.

The source is CSTR VCTK Corpus 0.92 under CC BY 4.0. Original VCTK audio and
transcript files remain in the local-only source tree and are never modified.
Reported source paths are relative to that source root.

## Stages

1. **Candidate indexing** - `scripts/prepare_vctk_speaker_pool.py` indexes the
   local corpus into `candidate_report.json` and `candidate_report.tsv`.
2. **Manual shortlist** - a human chooses speakers and their primary/backup
   utterances, recorded in `selected_speakers.json`.
3. **Reference materialisation** - `scripts/materialize_vctk_speaker_pool.py`
   converts the selected recordings into `references/` and writes
   `speaker_registry.json`.
4. **Same-text CosyVoice smoke** - `scripts/run_speaker_pool_smoke.py`
   synthesizes one shared neutral sentence with every speaker reference through
   the existing production synthesis path.
5. **Human review** - listen to the smoke output and fill in the generated
   `smoke_review.tsv`.
6. **Final active speaker pool** - decided only after that human review.
7. **Deterministic dialogue assignment** - a later stage; not implemented here.

Stages 1 to 4 are complete for the current shortlist. Stages 5 to 7 are not.

## Current shortlist status

- The 12 speakers in `selected_speakers.json` are a provisional smoke-test
  shortlist, not an accepted speaker pool.
- Primary and backup references can change after smoke testing and human
  listening; the shortlist is never reranked or replaced automatically.
- Speaker demographics are copied verbatim from VCTK `speaker-info.txt` through
  `candidate_report.json`. Nothing is inferred from audio.
- The shortlist spread across accent, region, and gender is experimental
  diversity for testing. It is not population representativeness and supports
  no demographic claim.
- No speaker is declared good, usable, or ranked by any script in this
  directory.

## Candidate indexing method

- Speakers `p280` and `p315` are excluded from v0.1 because of the documented
  recording/data issues. In particular, p315 has no transcript data.
- `{config["microphone"]}` is used consistently; recordings from other
  microphones are skipped.
- Matched references must be {config["min_duration_sec"]:.1f} to
  {config["max_duration_sec"]:.1f} seconds and contain at least
  {config["min_transcript_words"]} words. A permissive
  {config["max_transcript_words_per_sec"]:.1f}-words/second ceiling removes
  transcripts unusually long for their audio duration.
- Ranking favors duration near {config["ranking_duration_center_sec"]:.1f}
  seconds and word count near {config["ranking_word_count_center"]}, with stable
  path-based tie-breakers. These duration and transcript filters are heuristics,
  not perceptual-quality scores.
- Final primary and backup reference selection requires human listening. No
  demographic attributes are inferred from audio; only fields explicitly
  present in `speaker-info.txt` are reported.

## Reference materialisation

Regenerate the reference package from the local corpus with:

```bash
uv run python scripts/materialize_vctk_speaker_pool.py
```

- Every reference is WAV, mono, 16 kHz, signed 16-bit PCM, which is the prompt
  format the CosyVoice zero-shot path expects.
- Only channel, sample-rate, and bit-depth conversion is applied. There is no
  denoising, pitch shifting, speed change, added silence, telephone filtering,
  loudness enhancement, or change of speech content.
- Each written reference is verified for existence, non-zero size, sample rate,
  channel count, 16-bit PCM subtype, and duration preserved within a small
  resampling tolerance. A failed reference fails the whole command rather than
  silently skipping a selected speaker.
- `prompt_text` is read from the VCTK transcript file at materialisation time
  and preserved exactly; no prompt text is hard-coded in Python.

## Speaker identity and provenance

`speaker_id` (for example `spk_001`) is the corpus-facing identifier and encodes
no gender, accent, or region. `source_speaker_id` (for example `p248`) is
retained as provenance, so every reference traces deterministically:

`spk_001` -> `p248` -> `p248_112` -> original VCTK FLAC and transcript ->
materialised reference WAV (recorded with its SHA256).

## Speaker-pool smoke test

The smoke test renders one shared sentence per speaker through
`tts5703.tts_engine.synthesize_turn`, so it reuses the persistent CosyVoice
worker and the production zero-shot path. It maps a temporary in-memory smoke
role to each speaker's reference instead of adding permanent voice-map entries
to `config/config.yaml`. Requested acoustic conditions are deliberately neutral
(normal rate, no pauses, no arousal, affect, emotion, or paralinguistic events),
because the test targets speaker identity, not acoustic control.

`speaker_registry.json` stores the exact VCTK transcript as `prompt_text`, while
the CosyVoice3 zero-shot path expects that transcript prefixed with
`You are a helpful assistant.<|endofprompt|>`. The smoke runner therefore
applies that formatting only to the temporary model-facing config and records
`prompt_format: cosyvoice3_zero_shot` in its report; the registry transcript and
the source VCTK transcripts are never rewritten.

`smoke_results.json` is a smoke-test report only: elapsed time, audio duration,
and real-time factor are recorded for traceability and must not be read as
speaker-quality scores or used to rank speakers.

## Files

- `candidate_report.json`: complete indexing summary, exclusion/rejection
  counts, source metadata, and up to the configured top K candidates per
  eligible speaker.
- `candidate_report.tsv`: one row per reported candidate for manual review.
- `selected_speakers.json`: the frozen manual shortlist. It intentionally
  duplicates no transcript or demographic metadata; `candidate_report.json`
  remains the source of truth for those values.
- `speaker_registry.json`: generated reference package with provenance,
  transcripts, durations, sample rates, and reference SHA256 values.
- `references/*.wav`: generated reference audio. Not committed; regenerate it
  locally from VCTK.
- `smoke_runs/<run_id>/`: generated smoke output, results, and review sheet.
  Not committed.

## Licence and attribution

The source corpus is CC BY 4.0, so attribution and provenance must be retained
wherever this material or anything derived from it is shared. The reference WAVs
are derived works of CC BY 4.0 material; they are reproducibly regenerated from
the local corpus and are not committed. The VCTK corpus itself is never
committed to this repository.

## Citation

Yamagishi, Junichi; Veaux, Christophe; MacDonald, Kirsten. (2019).
*CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit
(version 0.92).* University of Edinburgh, CSTR. DOI: 10.7488/ds/2645
"""


def write_candidate_reports(
    output_dir: Path, report: dict[str, Any], demographic_fields: list[str]
) -> tuple[Path, Path, Path]:
    """Write deterministic JSON, TSV, and documentation outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "candidate_report.json"
    tsv_path = output_dir / "candidate_report.tsv"
    readme_path = output_dir / "README.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_tsv(tsv_path, report, demographic_fields)
    readme_path.write_text(_readme_text(report), encoding="utf-8")
    return json_path, tsv_path, readme_path


def prepare_speaker_pool(
    vctk_root: Path,
    output_dir: Path,
    *,
    mic: str = DEFAULT_MIC,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[dict[str, Any], tuple[Path, Path, Path]]:
    """Build and write candidate reports without copying source audio."""
    report, demographic_fields = build_candidate_report(
        vctk_root,
        mic=mic,
        min_duration=min_duration,
        max_duration=max_duration,
        top_k=top_k,
    )
    outputs = write_candidate_reports(output_dir, report, demographic_fields)
    return report, outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index local VCTK 0.92 reference-speech candidates"
    )
    parser.add_argument("--vctk-root", type=Path, default=DEFAULT_VCTK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mic", default=DEFAULT_MIC)
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION)
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report, outputs = prepare_speaker_pool(
            args.vctk_root,
            args.output_dir,
            mic=args.mic,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            top_k=args.top_k,
        )
    except (VCTKPreparationError, OSError) as error:
        print(f"VCTK preparation failed: {error}", file=sys.stderr)
        return 2

    summary = report["summary"]
    print(f"Speakers indexed: {summary['speakers_indexed']}")
    print(f"Eligible speakers: {summary['speakers_eligible']}")
    print(f"Matched utterances: {summary['matched_utterances']}")
    print(f"Candidate utterances: {summary['candidate_utterances']}")
    print(f"Speakers with candidates: {summary['speakers_with_candidates']}")
    for output in outputs:
        print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
