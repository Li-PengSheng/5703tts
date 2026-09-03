"""Synthesize one shared sentence with every selected speaker reference.

This is a speaker-identity smoke test, not an acoustic-control experiment and
not a speaker-quality score. Every speaker receives exactly the same text and
the same neutral acoustic conditions, and synthesis goes through the existing
production path (``tts5703.tts_engine.synthesize_turn``) so the persistent
CosyVoice worker is loaded once and reused for all speakers.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import soundfile as sf

from tts5703.config import load_config
from tts5703.tts_engine import describe_engine, get_engine, synthesize_turn
from tts5703.validate import NormalizedTurn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEAKER_POOL_DIR = PROJECT_ROOT / "data" / "speaker_pool" / "vctk_v0.1"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_REGISTRY = SPEAKER_POOL_DIR / "speaker_registry.json"
DEFAULT_OUTPUT_ROOT = SPEAKER_POOL_DIR / "smoke_runs"
RESULTS_FILENAME = "smoke_results.json"
REVIEW_FILENAME = "smoke_review.tsv"
REQUIRED_ENGINE = "cosyvoice"
RUN_ID_PREFIX = "cosyvoice3"
REFERENCE_ROLES = ("primary", "backup")
DEFAULT_REFERENCE_ROLE = "primary"
# A temporary role that exists only inside the per-speaker in-memory config, so
# config.yaml never accumulates permanent speaker-pool voice-map entries.
SMOKE_SPEAKER_ROLE = "speaker_pool_smoke"
SMOKE_TURN_ID = 1
SMOKE_LABEL = "normal"
DEFAULT_SMOKE_TEXT = (
    "I understand. Take your time and tell me what happened when you are ready."
)
COSYVOICE3_END_OF_PROMPT = "<|endofprompt|>"
COSYVOICE3_ZERO_SHOT_PROMPT_PREFIX = (
    f"You are a helpful assistant.{COSYVOICE3_END_OF_PROMPT}"
)
PROMPT_FORMAT = "cosyvoice3_zero_shot"
REQUESTED_ACOUSTIC_CONDITIONS: dict[str, Any] = {
    "rate": "normal",
    "pause_before_ms": 0,
    "pause_after_ms": 0,
    "arousal": None,
    "coarse_affect": None,
    "emotion": None,
    "paralinguistic_events": [],
}
HUMAN_REVIEW_COLUMNS = (
    "naturalness",
    "identity_preservation",
    "distinct_from_others",
    "artifacts",
    "decision",
    "notes",
)
REVIEW_COLUMNS = (
    "speaker_id",
    "source_speaker_id",
    "gender",
    "accent",
    "region",
    "source_utterance_id",
    "generated_audio",
    *HUMAN_REVIEW_COLUMNS,
)
TIMING_DEFINITION = (
    "elapsed_sec is end-to-end wall-clock time around synthesize_turn and may "
    "include IPC, frontend work, inference, and file writing; the first speaker "
    "additionally pays one-time CosyVoice worker startup"
)
RANKING_DISCLAIMER = (
    "Smoke-test report only. Speakers are not ranked, scored, or declared usable "
    "from elapsed time, duration, or RTF; that requires human listening."
)


class SpeakerSmokeError(ValueError):
    """Raised when the smoke run cannot be set up as specified."""


def parse_speaker_ids(value: str | None) -> list[str] | None:
    """Parse a comma-separated ``--speaker-ids`` value.

    ``None`` means every registry speaker. Duplicates and empty entries are
    rejected rather than silently dropped, so a mistyped list cannot shrink
    the run without being noticed.
    """
    if value is None:
        return None
    if not value.strip():
        raise SpeakerSmokeError(
            "speaker-ids must be a non-empty comma-separated list of speaker IDs"
        )
    speaker_ids: list[str] = []
    seen: set[str] = set()
    for raw in value.split(","):
        speaker_id = raw.strip()
        if not speaker_id:
            raise SpeakerSmokeError(f"speaker-ids contains an empty entry: {value!r}")
        if speaker_id in seen:
            raise SpeakerSmokeError(
                f"Duplicate speaker ID in --speaker-ids: {speaker_id}"
            )
        seen.add(speaker_id)
        speaker_ids.append(speaker_id)
    return speaker_ids


def select_registry_speakers(
    registry: dict[str, Any], speaker_ids: list[str] | None
) -> list[dict[str, Any]]:
    """Return the registry speakers to test, preserving the requested order.

    Unknown IDs fail here, before any synthesis starts. ``None`` keeps the
    registry order unchanged.
    """
    speakers = registry["speakers"]
    if speaker_ids is None:
        return list(speakers)
    if not speaker_ids:
        raise SpeakerSmokeError("speaker-ids must not be empty")
    seen: set[str] = set()
    cleaned: list[str] = []
    for speaker_id in speaker_ids:
        if not isinstance(speaker_id, str) or not speaker_id.strip():
            raise SpeakerSmokeError(
                "speaker-ids must be a non-empty comma-separated list of speaker IDs"
            )
        speaker_id = speaker_id.strip()
        if speaker_id in seen:
            raise SpeakerSmokeError(
                f"Duplicate speaker ID in --speaker-ids: {speaker_id}"
            )
        seen.add(speaker_id)
        cleaned.append(speaker_id)
    by_id = {
        entry["speaker_id"]: entry
        for entry in speakers
        if isinstance(entry.get("speaker_id"), str)
    }
    unknown = [speaker_id for speaker_id in cleaned if speaker_id not in by_id]
    if unknown:
        known = [entry["speaker_id"] for entry in speakers]
        raise SpeakerSmokeError(
            f"Unknown speaker ID(s): {', '.join(unknown)}. "
            f"Known speaker IDs: {', '.join(known)}"
        )
    return [by_id[speaker_id] for speaker_id in cleaned]


def load_registry(path: Path) -> dict[str, Any]:
    """Load the materialised speaker registry produced by the materialisation step."""
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SpeakerSmokeError(
            f"Speaker registry not found: {path}. Run "
            "scripts/materialize_vctk_speaker_pool.py first."
        ) from error
    except json.JSONDecodeError as error:
        raise SpeakerSmokeError(
            f"Speaker registry is not valid JSON ({path}): {error}"
        ) from error
    if not isinstance(registry, dict):
        raise SpeakerSmokeError(f"Speaker registry must be a JSON object: {path}")
    speakers = registry.get("speakers")
    if not isinstance(speakers, list) or not speakers:
        raise SpeakerSmokeError(f"Speaker registry lists no speakers: {path}")
    return registry


def require_cosyvoice(config: dict[str, Any], config_path: Path) -> str:
    """Require the production CosyVoice path; no other engine is smoke-tested here."""
    engine = get_engine(config)
    if engine != REQUIRED_ENGINE:
        raise SpeakerSmokeError(
            f"Speaker-pool smoke testing requires tts.engine == {REQUIRED_ENGINE!r}, "
            f"but {config_path} selects {engine!r}"
        )
    return engine


def smoke_turn(text: str) -> NormalizedTurn:
    """Build the single neutral turn every speaker renders identically."""
    return NormalizedTurn(
        turn_id=SMOKE_TURN_ID,
        speaker=SMOKE_SPEAKER_ROLE,
        text=text,
        label=SMOKE_LABEL,
        rate=REQUESTED_ACOUSTIC_CONDITIONS["rate"],
        pause_before_ms=REQUESTED_ACOUSTIC_CONDITIONS["pause_before_ms"],
        pause_after_ms=REQUESTED_ACOUSTIC_CONDITIONS["pause_after_ms"],
        arousal=REQUESTED_ACOUSTIC_CONDITIONS["arousal"],
        coarse_affect=REQUESTED_ACOUSTIC_CONDITIONS["coarse_affect"],
        emotion=REQUESTED_ACOUSTIC_CONDITIONS["emotion"],
        paralinguistic_events=list(
            REQUESTED_ACOUSTIC_CONDITIONS["paralinguistic_events"]
        ),
    )


def format_cosyvoice3_zero_shot_prompt_text(raw_transcript: str) -> str:
    """Format one raw reference transcript the way CosyVoice3 zero-shot expects.

    Fun-CosyVoice3-0.5B's zero-shot example prefixes the reference transcript
    with an assistant preamble terminated by the ``<|endofprompt|>`` marker its
    LLM requires. ``speaker_registry.json`` deliberately stores the exact VCTK
    transcript for provenance, so the marker is added here, at the model
    boundary only, and the registry is never rewritten.
    """
    if not isinstance(raw_transcript, str) or not raw_transcript.strip():
        raise SpeakerSmokeError(
            "The raw reference transcript must be a non-empty string"
        )
    if COSYVOICE3_END_OF_PROMPT in raw_transcript:
        raise SpeakerSmokeError(
            f"The raw reference transcript already contains "
            f"{COSYVOICE3_END_OF_PROMPT}, so CosyVoice3 zero-shot formatting would "
            f"double-prefix it: {raw_transcript!r}. The registry must store the "
            "exact VCTK transcript without any model-specific prompt formatting."
        )
    return f"{COSYVOICE3_ZERO_SHOT_PROMPT_PREFIX}{raw_transcript}"


def speaker_config(
    base_config: dict[str, Any], prompt_wav: Path, raw_prompt_text: str
) -> dict[str, Any]:
    """Copy the base config and point the temporary smoke role at one reference.

    The copy is the only model-facing configuration, so the CosyVoice3 prompt
    formatting is applied here and the raw registry transcript stays raw. Only
    the voice map changes, so every copy keeps the same worker start-up
    parameters and the cached CosyVoice worker is reused across speakers.
    """
    config = copy.deepcopy(base_config)
    config["tts"]["cosyvoice"]["voice_map"][SMOKE_SPEAKER_ROLE] = {
        "prompt_wav": str(prompt_wav),
        "prompt_text": format_cosyvoice3_zero_shot_prompt_text(raw_prompt_text),
    }
    return config


def _reference(entry: dict[str, Any], role: str, speaker_id: str) -> dict[str, Any]:
    reference = entry.get(f"{role}_reference")
    if not isinstance(reference, dict):
        raise SpeakerSmokeError(f"{speaker_id}: registry has no {role} reference")
    for field in ("prompt_wav", "prompt_text", "source_utterance_id"):
        value = reference.get(field)
        if not isinstance(value, str) or not value.strip():
            raise SpeakerSmokeError(
                f"{speaker_id}: {role} reference is missing {field}"
            )
    return reference


def _resolve_prompt_wav(reference_root: Path, prompt_wav: str) -> Path:
    path = Path(prompt_wav)
    resolved = path if path.is_absolute() else reference_root / path
    if not resolved.is_file():
        raise SpeakerSmokeError(
            f"Reference audio is missing: {resolved}. Re-run "
            "scripts/materialize_vctk_speaker_pool.py to regenerate it."
        )
    return resolved


def read_wav_metrics(path: Path) -> tuple[float, int]:
    """Return positive WAV duration and sample rate without loading audio data."""
    info = sf.info(path)
    duration = info.frames / info.samplerate if info.samplerate > 0 else 0.0
    if duration <= 0:
        raise ValueError(f"Generated WAV has invalid duration: {path}")
    return duration, int(info.samplerate)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _engine_description(config: dict[str, Any]) -> dict[str, Any]:
    described = describe_engine(config)
    return {
        field: described[field]
        for field in (
            "engine",
            "model",
            "model_dir",
            "repo_dir",
            "fp16",
            "load_trt",
            "load_vllm",
            "expected_sample_rate",
        )
        if field in described
    }


def _new_run_directory(output_root: Path, run_id: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    candidate = output_root / run_id
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{run_id}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _write_results(results: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_review_sheet(
    path: Path, registry: dict[str, Any], results: dict[str, Any], role: str
) -> None:
    """Write a listening sheet whose judgment columns stay empty for a human."""
    metadata_by_id = {
        entry["speaker_id"]: entry.get("source_metadata", {})
        for entry in registry["speakers"]
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, dialect="excel-tab")
        writer.writeheader()
        for result in results["speaker_results"]:
            metadata = metadata_by_id.get(result["speaker_id"], {})
            row = {
                "speaker_id": result["speaker_id"],
                "source_speaker_id": result["source_speaker_id"],
                "gender": metadata.get("gender", ""),
                "accent": metadata.get("accent", ""),
                "region": metadata.get("region", ""),
                "source_utterance_id": result["source_utterance_id"] or "",
                "generated_audio": result["generated_audio"] or "",
                **{column: "" for column in HUMAN_REVIEW_COLUMNS},
            }
            writer.writerow(row)


def _overall_status(speaker_results: list[dict[str, Any]]) -> str:
    successes = sum(result["success"] for result in speaker_results)
    if successes == len(speaker_results):
        return "complete"
    return "partial_failure" if successes else "failed"


async def _execute_smoke_run(
    *,
    speakers: list[dict[str, Any]],
    base_config: dict[str, Any],
    turn: NormalizedTurn,
    role: str,
    reference_root: Path,
    run_dir: Path,
    results: dict[str, Any],
    clock: Callable[[], float],
    utc_now: Callable[[], datetime],
) -> None:
    """Render each selected speaker, keeping failures visible in the report."""
    results_path = run_dir / RESULTS_FILENAME
    run_started = clock()
    for entry in speakers:
        speaker_id = entry["speaker_id"]
        result: dict[str, Any] = {
            "speaker_id": speaker_id,
            "source_speaker_id": entry.get("source_speaker_id"),
            "source_utterance_id": None,
            "prompt_wav": None,
            "prompt_text": None,
            "prompt_format": None,
            "generated_audio": None,
            "success": False,
            "elapsed_sec": None,
            "duration_sec": None,
            "real_time_factor": None,
            "error": None,
        }
        results["speaker_results"].append(result)
        try:
            reference = _reference(entry, role, speaker_id)
            result["source_utterance_id"] = reference["source_utterance_id"]
            result["prompt_wav"] = reference["prompt_wav"]
            result["prompt_text"] = reference["prompt_text"]
            prompt_wav = _resolve_prompt_wav(reference_root, reference["prompt_wav"])
            config = speaker_config(base_config, prompt_wav, reference["prompt_text"])
            result["prompt_format"] = PROMPT_FORMAT
            started = clock()
            output_path = await synthesize_turn(turn, run_dir / speaker_id, config)
            result["elapsed_sec"] = round(clock() - started, 6)
            result["generated_audio"] = output_path.relative_to(run_dir).as_posix()
            duration_sec, _ = read_wav_metrics(output_path)
            result["duration_sec"] = round(duration_sec, 6)
            result["real_time_factor"] = round(result["elapsed_sec"] / duration_sec, 6)
            result["success"] = True
        except Exception as error:  # noqa: BLE001 - one failure must not hide others
            result["error"] = f"{type(error).__name__}: {error}"
        _write_results(results, results_path)

    results["successful_speakers"] = sum(
        result["success"] for result in results["speaker_results"]
    )
    results["failed_speakers"] = (
        len(results["speaker_results"]) - results["successful_speakers"]
    )
    results["status"] = _overall_status(results["speaker_results"])
    results["completed_at_utc"] = utc_now().isoformat()
    results["total_wall_sec"] = round(clock() - run_started, 6)
    _write_results(results, results_path)


def run_speaker_pool_smoke(
    *,
    config_path: Path = DEFAULT_CONFIG,
    registry_path: Path = DEFAULT_REGISTRY,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    text: str = DEFAULT_SMOKE_TEXT,
    role: str = DEFAULT_REFERENCE_ROLE,
    reference_root: Path = PROJECT_ROOT,
    run_id: str | None = None,
    speaker_ids: list[str] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    utc_now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[dict[str, Any], Path]:
    """Run the same-text speaker smoke test over selected registry speakers."""
    if role not in REFERENCE_ROLES:
        raise SpeakerSmokeError(
            f"Unsupported reference role {role!r}; supported roles: "
            f"{list(REFERENCE_ROLES)}"
        )
    if not text.strip():
        raise SpeakerSmokeError("The smoke-test sentence must not be empty")
    config = load_config(config_path)
    engine = require_cosyvoice(config, config_path)
    registry = load_registry(registry_path)
    speakers = select_registry_speakers(registry, speaker_ids)
    turn = smoke_turn(text)

    started_at = utc_now()
    resolved_run_id = run_id or f"{RUN_ID_PREFIX}_{started_at:%Y%m%dT%H%M%SZ}"
    run_dir = _new_run_directory(output_root, resolved_run_id)
    results: dict[str, Any] = {
        "run_id": run_dir.name,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": None,
        "status": "running",
        "purpose": (
            "Speaker identity and cloning smoke test using one shared sentence per "
            "speaker."
        ),
        "engine": engine,
        "engine_description": _engine_description(config),
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "git_commit": _git_commit(),
        },
        "speaker_registry": {
            "path": registry_path.as_posix(),
            "speaker_pool_version": registry.get("speaker_pool_version"),
            "status": registry.get("status"),
        },
        "reference_role": role,
        "test_sentence": text,
        "label": SMOKE_LABEL,
        "prompt_formatting": {
            "prompt_format": PROMPT_FORMAT,
            "model_facing_prefix": COSYVOICE3_ZERO_SHOT_PROMPT_PREFIX,
            "applied_to": "model_facing_prompt_text_only",
            "note": (
                "Per-speaker prompt_text is the exact raw VCTK reference transcript "
                "from speaker_registry.json. The model-facing prompt text is "
                "model_facing_prefix immediately followed by that raw transcript; "
                "the registry and the source transcripts are never rewritten."
            ),
        },
        "requested_acoustic_conditions": copy.deepcopy(REQUESTED_ACOUSTIC_CONDITIONS),
        "timing_definition": TIMING_DEFINITION,
        "ranking_disclaimer": RANKING_DISCLAIMER,
        "expected_speaker_ids": [entry["speaker_id"] for entry in speakers],
        "speaker_count": len(speakers),
        "successful_speakers": 0,
        "failed_speakers": 0,
        "speaker_results": [],
    }
    _write_results(results, run_dir / RESULTS_FILENAME)
    asyncio.run(
        _execute_smoke_run(
            speakers=speakers,
            base_config=config,
            turn=turn,
            role=role,
            reference_root=reference_root,
            run_dir=run_dir,
            results=results,
            clock=clock,
            utc_now=utc_now,
        )
    )
    write_review_sheet(run_dir / REVIEW_FILENAME, registry, results, role)
    return results, run_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize one shared sentence with every selected speaker reference "
            "through the production CosyVoice path"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--text", default=DEFAULT_SMOKE_TEXT)
    parser.add_argument(
        "--reference", choices=REFERENCE_ROLES, default=DEFAULT_REFERENCE_ROLE
    )
    parser.add_argument("--reference-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--speaker-ids",
        default=None,
        metavar="ID,ID,...",
        help=(
            "Comma-separated speaker IDs to test, in that order. "
            "When omitted, every registry speaker is tested."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        results, run_dir = run_speaker_pool_smoke(
            config_path=args.config,
            registry_path=args.registry,
            output_root=args.output_root,
            text=args.text,
            role=args.reference,
            reference_root=args.reference_root,
            run_id=args.run_id,
            speaker_ids=parse_speaker_ids(args.speaker_ids),
        )
    except Exception as error:  # noqa: BLE001 - concise CLI boundary
        print(
            f"Speaker pool smoke setup failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2

    print(f"Run directory: {run_dir}")
    print(f"Reference role: {results['reference_role']}")
    print(
        f"Successful speakers: {results['successful_speakers']}"
        f"/{results['speaker_count']}"
    )
    print(f"Status: {results['status']}")
    for result in results["speaker_results"]:
        if not result["success"]:
            print(
                f"Failed: {result['speaker_id']} ({result['source_speaker_id']}): "
                f"{result['error']}",
                file=sys.stderr,
            )
    print(f"Results: {run_dir / RESULTS_FILENAME}")
    print(f"Review sheet: {run_dir / REVIEW_FILENAME}")
    return 0 if results["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
