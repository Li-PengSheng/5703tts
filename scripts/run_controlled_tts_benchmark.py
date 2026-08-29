"""Run the controlled TTS benchmark through a supported production engine."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import soundfile as sf

from tts5703.assemble import TurnTiming, assemble_dialogue
from tts5703.config import load_config
from tts5703.engine_capabilities import UnknownEngineCapabilityError
from tts5703.engine_capabilities import (
    engine_capabilities as declared_engine_capabilities,
)
from tts5703.engine_capabilities import (
    ignored_requested_controls as declared_ignored_controls,
)
from tts5703.engine_capabilities import (
    requested_acoustic_spec as declared_requested_acoustic_spec,
)
from tts5703.tts_engine import (
    build_cosyvoice_request,
    describe_engine,
    get_engine,
    rate_to_kokoro_speed,
    synthesize_turn,
)
from tts5703.validate import NormalizedTurn, load_and_validate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_FIXTURE = PROJECT_ROOT / "data" / "benchmark" / "benchmark_fixture_v0.1.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "benchmark" / "benchmark_manifest_v0.1.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "benchmark" / "runs"
RESULTS_FILENAME = "benchmark_results.json"
SUPPORTED_BENCHMARK_ENGINES = {"cosyvoice", "kokoro"}
RUN_ID_PREFIXES = {"cosyvoice": "cosyvoice3", "kokoro": "kokoro"}


class BenchmarkDesignError(ValueError):
    """Raised before synthesis when fixture and manifest controls disagree."""


def benchmark_engine(config: dict[str, Any]) -> str:
    """Return an explicitly supported controlled-benchmark engine."""
    engine = get_engine(config)
    if engine not in SUPPORTED_BENCHMARK_ENGINES:
        raise BenchmarkDesignError(
            f"Unsupported controlled benchmark engine: {engine!r}; supported engines: "
            f"{sorted(SUPPORTED_BENCHMARK_ENGINES)}"
        )
    return engine


def engine_capabilities(engine: str) -> dict[str, dict[str, str]]:
    """Return the production capability declaration for one benchmark engine.

    The benchmark deliberately owns no capability table of its own; production
    metadata and benchmark reporting must not be able to drift apart.
    """
    try:
        return declared_engine_capabilities(engine)
    except UnknownEngineCapabilityError as error:
        raise BenchmarkDesignError(
            f"No controlled benchmark capabilities declared for engine {engine!r}"
        ) from error


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise BenchmarkDesignError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise BenchmarkDesignError(f"Expected a JSON object in {path}")
    return value


def _validate_controlled_group(
    group_id: str, target: str, turns: list[dict[str, Any]]
) -> None:
    """Require one manipulated acoustic field and otherwise identical turns."""
    if len(turns) < 2:
        turn_ids = [turn["turn_id"] for turn in turns]
        raise BenchmarkDesignError(
            f"Group {group_id!r} target acoustic field {target!r} does not vary "
            f"across turns {turn_ids}"
        )
    baseline = turns[0]
    baseline_id = baseline["turn_id"]
    for turn in turns[1:]:
        turn_id = turn["turn_id"]
        for field in ("text", "speaker", "label"):
            if turn[field] != baseline[field]:
                raise BenchmarkDesignError(
                    f"Group {group_id!r} is confounded: {field} differs "
                    f"between turns {baseline_id} and {turn_id}"
                )

        baseline_spec = baseline["acoustic_spec"]
        turn_spec = turn["acoustic_spec"]
        non_target_fields = (set(baseline_spec) | set(turn_spec)) - {target}
        missing = object()
        for field in sorted(non_target_fields):
            if baseline_spec.get(field, missing) != turn_spec.get(field, missing):
                raise BenchmarkDesignError(
                    f"Group {group_id!r} is confounded: non-target acoustic "
                    f"field {field!r} differs between turns {baseline_id} and {turn_id}"
                )

    target_values = [turn["acoustic_spec"][target] for turn in turns]
    if not any(value != target_values[0] for value in target_values[1:]):
        turn_ids = [turn["turn_id"] for turn in turns]
        raise BenchmarkDesignError(
            f"Group {group_id!r} target acoustic field {target!r} does not vary "
            f"across turns {turn_ids}"
        )


def build_manifest_lookup(
    fixture: dict[str, Any], manifest: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    """Validate one-group-per-turn membership and return turn design metadata."""
    raw_turns = fixture.get("turns")
    groups = manifest.get("groups")
    if not isinstance(raw_turns, list) or not isinstance(groups, list):
        raise BenchmarkDesignError("Fixture turns and manifest groups must be arrays")

    turns_by_id = {turn["turn_id"]: turn for turn in raw_turns}
    lookup: dict[int, dict[str, Any]] = {}
    controlled_groups: list[tuple[str, str, list[dict[str, Any]]]] = []
    group_ids: set[str] = set()
    for group in groups:
        group_id = group.get("group_id")
        target = group.get("target_variable")
        turn_ids = group.get("turn_ids")
        conditions = group.get("expected_conditions")
        if not isinstance(group_id, str) or not group_id.strip():
            raise BenchmarkDesignError(
                "Every manifest group needs a non-empty string group_id"
            )
        if group_id in group_ids:
            raise BenchmarkDesignError(f"Duplicate benchmark group_id: {group_id!r}")
        group_ids.add(group_id)
        if not isinstance(target, str):
            raise BenchmarkDesignError(
                f"Manifest group {group_id!r} needs a string target_variable"
            )
        if not isinstance(turn_ids, list) or not isinstance(conditions, list):
            raise BenchmarkDesignError(
                f"Manifest group {group_id!r} needs turn_ids and expected_conditions"
            )
        if len(turn_ids) != len(conditions):
            raise BenchmarkDesignError(
                f"Manifest group {group_id!r} has mismatched turn and condition counts"
            )

        group_turns = []
        for turn_id, expected_condition in zip(turn_ids, conditions, strict=True):
            if turn_id not in turns_by_id:
                raise BenchmarkDesignError(
                    f"Manifest group {group_id!r} references unknown turn {turn_id}"
                )
            if turn_id in lookup:
                raise BenchmarkDesignError(
                    f"Turn {turn_id} belongs to more than one manifest group"
                )
            acoustic_spec = turns_by_id[turn_id].get("acoustic_spec", {})
            if target not in acoustic_spec:
                raise BenchmarkDesignError(
                    f"Turn {turn_id} has no acoustic field {target!r}"
                )
            actual_condition = acoustic_spec[target]
            if actual_condition != expected_condition:
                raise BenchmarkDesignError(
                    f"Turn {turn_id} manifest condition {expected_condition!r} "
                    f"does not match fixture value {actual_condition!r}"
                )
            lookup[turn_id] = {
                "group_id": group_id,
                "target_variable": target,
                "target_condition": actual_condition,
            }
            group_turns.append(turns_by_id[turn_id])
        controlled_groups.append((group_id, target, group_turns))

    missing = sorted(set(turns_by_id) - set(lookup))
    if missing:
        raise BenchmarkDesignError(
            f"Fixture turns missing manifest group membership: {missing}"
        )
    for group_id, target, group_turns in controlled_groups:
        _validate_controlled_group(group_id, target, group_turns)
    return lookup


def calculate_rtf(generation_elapsed_sec: float, audio_duration_sec: float) -> float:
    """Calculate end-to-end wall-clock real-time factor."""
    if audio_duration_sec <= 0:
        raise ValueError("audio duration must be greater than zero for RTF")
    return generation_elapsed_sec / audio_duration_sec


def read_wav_metrics(path: Path) -> tuple[float, int]:
    """Return positive WAV duration and sample rate without loading audio data."""
    info = sf.info(path)
    duration = info.frames / info.samplerate if info.samplerate > 0 else 0.0
    if duration <= 0:
        raise ValueError(f"Generated WAV has invalid duration: {path}")
    return duration, info.samplerate


def requested_acoustic_spec(turn: NormalizedTurn) -> dict[str, Any]:
    return declared_requested_acoustic_spec(turn)


def effective_cosyvoice_controls(
    turn: NormalizedTurn, config: dict[str, Any], output_path: Path
) -> dict[str, Any]:
    """Trace the exact production mapping without reimplementing it."""
    voice = config["tts"]["cosyvoice"]["voice_map"][turn.speaker]
    request = build_cosyvoice_request(
        turn,
        voice["prompt_text"],
        Path(voice["prompt_wav"]),
        output_path,
    )
    return {
        "speed": request["speed"],
        "mode": request["mode"],
        "instruction": request.get("instruction"),
    }


def effective_controls_for_turn(
    engine: str,
    turn: NormalizedTurn,
    config: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Trace the exact backend-boundary controls used by production synthesis."""
    if engine == "cosyvoice":
        return effective_cosyvoice_controls(turn, config, output_path)
    if engine == "kokoro":
        kokoro = config["tts"]["kokoro"]
        return {
            "speed": rate_to_kokoro_speed(turn.rate),
            "voice": kokoro["voice_map"][turn.speaker],
            "ignored_requested_controls": declared_ignored_controls(
                engine, requested_acoustic_spec(turn)
            ),
        }
    raise BenchmarkDesignError(
        f"No effective-control trace is available for engine {engine!r}"
    )


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


def environment_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    engine = benchmark_engine(config)
    engine_info = describe_engine(config)
    snapshot = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "git_commit": _git_commit(),
        "model": engine_info["model"],
    }
    if engine == "cosyvoice":
        cosyvoice = config["tts"]["cosyvoice"]
        snapshot.update(
            {
                "model_dir": cosyvoice.get("model_dir", "models/Fun-CosyVoice3-0.5B"),
                "fp16": cosyvoice.get("fp16", True),
                "load_trt": cosyvoice.get("load_trt", False),
                "load_vllm": cosyvoice.get("load_vllm", False),
            }
        )
    else:
        kokoro = config["tts"]["kokoro"]
        snapshot.update(
            {
                "lang_code": kokoro["lang_code"],
                "sample_rate": kokoro["sample_rate"],
                "device": kokoro.get("device"),
                "voice_map": dict(kokoro["voice_map"]),
            }
        )
    return snapshot


def control_scope(capabilities: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    """Summarise declared capabilities while retaining the existing result field."""
    return {
        "model_controls": [
            field
            for field, value in capabilities.items()
            if value["support"] in {"model_control", "provisional_model_control"}
        ],
        "pipeline_timing_controls": [
            field
            for field, value in capabilities.items()
            if value["support"] == "pipeline_timing"
        ],
        "unsupported_controls": [
            field
            for field, value in capabilities.items()
            if value["support"] == "unsupported"
        ],
    }


def _warmup_turn_id(manifest: dict[str, Any]) -> int:
    rate_groups = [
        group for group in manifest["groups"] if group.get("target_variable") == "rate"
    ]
    if len(rate_groups) != 1:
        raise BenchmarkDesignError("Manifest must contain exactly one rate group")
    turn_id = rate_groups[0].get("baseline_turn_id")
    if turn_id not in rate_groups[0].get("turn_ids", []):
        raise BenchmarkDesignError("Rate baseline turn must belong to the rate group")
    return turn_id


def _new_run_directory(output_root: Path, run_id: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    candidate = output_root / run_id
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{run_id}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    (candidate / "turns").mkdir()
    return candidate


def _write_results(results: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _failure(stage: str, error: Exception, turn_id: int | None = None) -> dict:
    failure = {
        "stage": stage,
        "error_type": type(error).__name__,
        "message": str(error),
    }
    if turn_id is not None:
        failure["turn_id"] = turn_id
    return failure


def build_group_summaries(
    manifest: dict[str, Any],
    turn_results: list[dict[str, Any]],
    capabilities: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Build descriptive diagnostics without assigning acoustic fidelity scores."""
    results_by_id = {result["turn_id"]: result for result in turn_results}
    summaries: dict[str, Any] = {}
    for group in manifest["groups"]:
        observations = []
        for turn_id in group["turn_ids"]:
            result = results_by_id.get(turn_id)
            if result is None:
                continue
            observations.append(
                {
                    "turn_id": turn_id,
                    "condition": result["target_condition"],
                    "generation_success": True,
                    "target_control_support": result["target_control_support"],
                    "audio_duration_sec": result["audio_duration_sec"],
                    "real_time_factor": result["real_time_factor"],
                    "effective_controls": result["effective_controls"],
                }
            )
        target = group["target_variable"]
        try:
            support_type = capabilities[target]["support"]
        except KeyError as error:
            raise BenchmarkDesignError(
                f"No capability is declared for target variable {target!r}"
            ) from error
        control_supported = support_type != "unsupported"
        summary: dict[str, Any] = {
            "target_variable": target,
            "control_supported": control_supported,
            "support_type": support_type,
            "observations": observations,
        }
        if not control_supported:
            summary["requested_target_control_ignored"] = True
            summary["controllability_conclusion"] = None
            summary["interpretation"] = (
                "Audio was generated, but the selected backend does not consume this "
                "requested acoustic control; this group is not evidence of "
                "controllability."
            )
        if target == "rate":
            durations = {
                observation["condition"]: observation["audio_duration_sec"]
                for observation in observations
            }
            direction_matches = None
            if {"slow", "normal", "fast"}.issubset(durations):
                direction_matches = (
                    durations["slow"] > durations["normal"] > durations["fast"]
                )
            summary["rate_direction_check"] = {
                "expected": "slow duration > normal duration > fast duration",
                "matches_expected_direction": direction_matches,
                "diagnostic_only": True,
            }
        summaries[group["group_id"]] = summary
    return summaries


def build_pause_diagnostic(
    turns: list[NormalizedTurn], timings: list[TurnTiming], audio_duration_ms: int
) -> dict[str, Any]:
    """Confirm assembly gaps reflect requested pipeline pause controls."""
    checks = []
    previous_end_ms = 0
    previous_pause_after_ms = 0
    for turn, timing in zip(turns, timings, strict=True):
        actual_gap_ms = round(timing.start_sec * 1000) - previous_end_ms
        expected_gap_ms = previous_pause_after_ms + turn.pause_before_ms
        checks.append(
            {
                "turn_id": turn.turn_id,
                "pause_before_ms": turn.pause_before_ms,
                "previous_pause_after_ms": previous_pause_after_ms,
                "expected_gap_before_speech_ms": expected_gap_ms,
                "actual_gap_before_speech_ms": actual_gap_ms,
                "matches": abs(actual_gap_ms - expected_gap_ms) <= 1,
            }
        )
        previous_end_ms = round(timing.end_sec * 1000)
        previous_pause_after_ms = turn.pause_after_ms

    expected_total_ms = previous_end_ms + previous_pause_after_ms
    total_matches = abs(audio_duration_ms - expected_total_ms) <= 1
    return {
        "control_type": "pipeline_timing",
        "turn_gaps": checks,
        "expected_total_duration_ms": expected_total_ms,
        "actual_total_duration_ms": audio_duration_ms,
        "passed": all(check["matches"] for check in checks) and total_matches,
    }


async def _execute_benchmark(
    *,
    engine: str,
    capabilities: dict[str, dict[str, str]],
    turns: list[NormalizedTurn],
    manifest: dict[str, Any],
    lookup: dict[int, dict[str, Any]],
    config: dict[str, Any],
    run_dir: Path,
    results: dict[str, Any],
    warmup_enabled: bool,
    clock: Callable[[], float],
    utc_now: Callable[[], datetime],
) -> None:
    results_path = run_dir / RESULTS_FILENAME
    turn_paths: dict[int, Path] = {}
    turns_by_id = {turn.turn_id: turn for turn in turns}
    total_started = clock()

    if warmup_enabled:
        warmup_turn_id = _warmup_turn_id(manifest)
        warmup_dir = run_dir / "warmup"
        warmup_dir.mkdir()
        warmup_started = clock()
        try:
            warmup_path = await synthesize_turn(
                turns_by_id[warmup_turn_id], warmup_dir, config
            )
        except Exception as error:  # noqa: BLE001 - persist experiment failure
            results["warmup"].update(
                {
                    "turn_id": warmup_turn_id,
                    "elapsed_sec": round(clock() - warmup_started, 6),
                }
            )
            results["failure"] = _failure("warmup", error, warmup_turn_id)
            results["status"] = "failed"
            results["completed_at_utc"] = utc_now().isoformat()
            results["total_benchmark_wall_sec"] = round(clock() - total_started, 6)
            _write_results(results, results_path)
            return
        results["warmup"].update(
            {
                "turn_id": warmup_turn_id,
                "elapsed_sec": round(clock() - warmup_started, 6),
                "output_path": warmup_path.relative_to(run_dir).as_posix(),
            }
        )

    for turn in turns:
        output_hint = run_dir / "turns" / f"turn_{turn.turn_id:03d}.wav"
        generation_started = clock()
        try:
            output_path = await synthesize_turn(turn, run_dir / "turns", config)
            generation_elapsed = clock() - generation_started
            audio_duration, sample_rate = read_wav_metrics(output_path)
            real_time_factor = calculate_rtf(generation_elapsed, audio_duration)
        except Exception as error:  # noqa: BLE001 - persist partial observations
            results["failure"] = _failure("timed_turn", error, turn.turn_id)
            results["status"] = "failed"
            results["completed_at_utc"] = utc_now().isoformat()
            results["total_benchmark_wall_sec"] = round(clock() - total_started, 6)
            results["group_summaries"] = build_group_summaries(
                manifest, results["turn_results"], capabilities
            )
            _write_results(results, results_path)
            return

        design = lookup[turn.turn_id]
        controls = effective_controls_for_turn(engine, turn, config, output_hint)
        target_control_support = capabilities[design["target_variable"]]["support"]
        results["turn_results"].append(
            {
                "turn_id": turn.turn_id,
                "group_id": design["group_id"],
                "target_variable": design["target_variable"],
                "target_condition": design["target_condition"],
                "target_control_support": target_control_support,
                "speaker": turn.speaker,
                "text": turn.text,
                "label": turn.label,
                "requested_acoustic_spec": requested_acoustic_spec(turn),
                "effective_controls": controls,
                "generation_elapsed_sec": round(generation_elapsed, 6),
                "worker_elapsed_sec": None,
                "audio_duration_sec": round(audio_duration, 6),
                "real_time_factor": round(real_time_factor, 6),
                "sample_rate": sample_rate,
                "output_path": output_path.relative_to(run_dir).as_posix(),
            }
        )
        turn_paths[turn.turn_id] = output_path
        _write_results(results, results_path)

    try:
        assembled, timings = assemble_dialogue(turns, turn_paths, config)
        clean_path = run_dir / "benchmark_clean.wav"
        assembled.export(clean_path, format="wav")
        results["assembly"] = {
            "output_path": clean_path.relative_to(run_dir).as_posix(),
            "timings": [
                {
                    "turn_id": timing.turn_id,
                    "speech_start_sec": timing.start_sec,
                    "speech_end_sec": timing.end_sec,
                }
                for timing in timings
            ],
            "pause_diagnostic": build_pause_diagnostic(turns, timings, len(assembled)),
        }
    except Exception as error:  # noqa: BLE001 - preserve completed turn outputs
        results["failure"] = _failure("assembly", error)
        results["status"] = "failed"
        results["completed_at_utc"] = utc_now().isoformat()
        results["total_benchmark_wall_sec"] = round(clock() - total_started, 6)
        results["group_summaries"] = build_group_summaries(
            manifest, results["turn_results"], capabilities
        )
        _write_results(results, results_path)
        return

    results["group_summaries"] = build_group_summaries(
        manifest, results["turn_results"], capabilities
    )
    results["status"] = "complete"
    results["completed_at_utc"] = utc_now().isoformat()
    results["total_benchmark_wall_sec"] = round(clock() - total_started, 6)
    _write_results(results, results_path)


def run_benchmark(
    *,
    config_path: Path = DEFAULT_CONFIG,
    fixture_path: Path = DEFAULT_FIXTURE,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    warmup_enabled: bool = True,
    run_id: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
    utc_now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[dict[str, Any], Path]:
    """Validate, render sequentially, assemble, and persist one benchmark run."""
    config = load_config(config_path)
    engine = benchmark_engine(config)
    capabilities = engine_capabilities(engine)
    engine_info = describe_engine(config)
    fixture = _load_json(fixture_path)
    manifest = _load_json(manifest_path)
    dialogue = load_and_validate(fixture_path, config)
    lookup = build_manifest_lookup(fixture, manifest)
    unknown_targets = sorted(
        {
            design["target_variable"]
            for design in lookup.values()
            if design["target_variable"] not in capabilities
        }
    )
    if unknown_targets:
        raise BenchmarkDesignError(
            f"No {engine} capability is declared for target variables: "
            f"{unknown_targets}"
        )
    if warmup_enabled:
        warmup_turn_id = _warmup_turn_id(manifest)
        if warmup_turn_id not in lookup:
            raise BenchmarkDesignError(
                "Warm-up turn is not part of the benchmark design"
            )

    started_at = utc_now()
    resolved_run_id = run_id or (
        f"{RUN_ID_PREFIXES[engine]}_{started_at:%Y%m%dT%H%M%SZ}"
    )
    run_dir = _new_run_directory(output_root, resolved_run_id)
    results = {
        "benchmark_version": manifest.get("benchmark_version"),
        "engine": engine,
        "model": engine_info["model"],
        "fixture": fixture_path.name,
        "manifest": manifest_path.name,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": None,
        "status": "running",
        "timing_definition": (
            "generation_elapsed_sec is end-to-end per-turn wall-clock time around "
            "synthesize_turn and may include IPC, frontend work, inference, and file writing"
        ),
        "control_capabilities": capabilities,
        "control_scope": control_scope(capabilities),
        "warmup": {
            "enabled": warmup_enabled,
            "turn_id": None,
            "elapsed_sec": None,
        },
        "environment": environment_snapshot(config),
        "turn_results": [],
        "group_summaries": {},
        "assembly": None,
        "failure": None,
    }
    _write_results(results, run_dir / RESULTS_FILENAME)
    asyncio.run(
        _execute_benchmark(
            engine=engine,
            capabilities=capabilities,
            turns=dialogue.turns,
            manifest=manifest,
            lookup=lookup,
            config=config,
            run_dir=run_dir,
            results=results,
            warmup_enabled=warmup_enabled,
            clock=clock,
            utc_now=utc_now,
        )
    )
    return results, run_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the controlled TTS benchmark through CosyVoice or Kokoro"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        results, run_dir = run_benchmark(
            config_path=args.config,
            fixture_path=args.fixture,
            manifest_path=args.manifest,
            output_root=args.output_root,
            warmup_enabled=not args.no_warmup,
        )
    except Exception as error:  # noqa: BLE001 - concise CLI boundary
        print(
            f"Benchmark setup failed: {type(error).__name__}: {error}", file=sys.stderr
        )
        return 2

    successful = len(results["turn_results"])
    total = sum(len(group["turn_ids"]) for group in _load_json(args.manifest)["groups"])
    print(f"Run directory: {run_dir}")
    print(f"Successful turns: {successful}/{total}")
    print(f"Total benchmark wall time: {results['total_benchmark_wall_sec']:.3f}s")
    if results["status"] == "complete":
        rtfs = [result["real_time_factor"] for result in results["turn_results"]]
        print(f"Mean RTF: {statistics.fmean(rtfs):.3f}")
        print(f"Median RTF: {statistics.median(rtfs):.3f}")
    else:
        print(f"Benchmark failed: {results['failure']}", file=sys.stderr)
    print(f"Results: {run_dir / RESULTS_FILENAME}")
    return 0 if results["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
