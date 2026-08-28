"""Measure lightweight acoustic descriptors for a completed benchmark run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ANALYSIS_FILENAME = "benchmark_acoustic_analysis.json"
FOCUS_TARGETS = ("rate", "arousal", "coarse_affect")
SILENCE_FRAME_MS = 20.0
SILENCE_THRESHOLD_DBFS = -40.0
SILENCE_THRESHOLD_AMPLITUDE = 10 ** (SILENCE_THRESHOLD_DBFS / 20)
DESCRIPTIVE_ONLY_NOTE = (
    "Descriptive acoustic difference only; requires later validation."
)


class AcousticAnalysisError(ValueError):
    """Raised when a benchmark result cannot be analysed safely."""


def _load_results(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AcousticAnalysisError(f"Benchmark results file not found: {path}")
    try:
        results = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AcousticAnalysisError(
            f"Invalid benchmark results JSON: {error}"
        ) from error
    if not isinstance(results, dict) or not isinstance(
        results.get("turn_results"), list
    ):
        raise AcousticAnalysisError(
            "Benchmark results must contain a turn_results array"
        )
    return results


def resolve_wav_path(results_path: Path, output_path: str) -> Path:
    """Resolve a result WAV reference relative to its benchmark run directory."""
    wav_path = Path(output_path)
    if not wav_path.is_absolute():
        wav_path = results_path.parent / wav_path
    if not wav_path.is_file():
        raise AcousticAnalysisError(f"Referenced WAV file not found: {wav_path}")
    return wav_path


def measure_wav(path: Path) -> dict[str, Any]:
    """Return duration, level, peak, and frame-based silence descriptors."""
    try:
        samples, sample_rate = sf.read(
            path, dtype="float64", always_2d=True, fill_value=0.0
        )
    except (OSError, RuntimeError, sf.LibsndfileError) as error:
        raise AcousticAnalysisError(
            f"Unable to read WAV file {path}: {error}"
        ) from error

    frame_count, channel_count = samples.shape
    if sample_rate <= 0 or frame_count <= 0:
        raise AcousticAnalysisError(f"WAV has invalid duration: {path}")

    duration_sec = frame_count / sample_rate
    rms_energy = math.sqrt(float(np.mean(np.square(samples))))
    peak_amplitude = float(np.max(np.abs(samples)))

    silence_frame_length = max(1, round(sample_rate * SILENCE_FRAME_MS / 1000))
    silent_frames = 0
    total_frames = 0
    for start in range(0, frame_count, silence_frame_length):
        frame = samples[start : start + silence_frame_length]
        frame_rms = math.sqrt(float(np.mean(np.square(frame))))
        silent_frames += frame_rms < SILENCE_THRESHOLD_AMPLITUDE
        total_frames += 1

    return {
        "audio_duration_sec": round(duration_sec, 6),
        "rms_energy": round(rms_energy, 9),
        "peak_amplitude": round(peak_amplitude, 9),
        "silence_proportion": round(silent_frames / total_frames, 6),
        "sample_rate": sample_rate,
        "channels": channel_count,
        "f0_metrics": None,
    }


def build_focus_group_lookup(
    turn_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Group the three model-control comparisons by their target variable."""
    lookup: dict[str, dict[str, Any]] = {}
    for turn in turn_results:
        target = turn.get("target_variable")
        if target not in FOCUS_TARGETS:
            continue
        group_id = turn.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise AcousticAnalysisError(
                f"Turn {turn.get('turn_id')} has no valid benchmark group_id"
            )
        group = lookup.setdefault(target, {"group_id": group_id, "turn_results": []})
        if group["group_id"] != group_id:
            raise AcousticAnalysisError(
                f"Target variable {target!r} occurs in multiple benchmark groups"
            )
        group["turn_results"].append(turn)

    missing = [target for target in FOCUS_TARGETS if target not in lookup]
    if missing:
        raise AcousticAnalysisError(
            f"Benchmark results are missing analysis groups: {missing}"
        )
    return lookup


def _conditions_by_target(
    target: str,
    group: dict[str, Any],
    metrics_by_turn_id: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    conditions: dict[str, dict[str, Any]] = {}
    for turn in group["turn_results"]:
        turn_id = turn.get("turn_id")
        condition = str(turn.get("target_condition"))
        if turn_id not in metrics_by_turn_id:
            raise AcousticAnalysisError(
                f"No acoustic metrics were measured for benchmark turn {turn_id}"
            )
        if condition in conditions:
            raise AcousticAnalysisError(
                f"Group {group['group_id']!r} has duplicate condition {condition!r}"
            )
        metrics = metrics_by_turn_id[turn_id]
        conditions[condition] = {
            "turn_id": turn_id,
            "audio_duration_sec": metrics["audio_duration_sec"],
            "rms_energy": metrics["rms_energy"],
            "peak_amplitude": metrics["peak_amplitude"],
            "silence_proportion": metrics["silence_proportion"],
            "f0_metrics": metrics["f0_metrics"],
        }

    required = {
        "rate": {"slow", "normal", "fast"},
        "arousal": {"low", "high"},
        "coarse_affect": {"neutral", "distressed"},
    }[target]
    if set(conditions) != required:
        raise AcousticAnalysisError(
            f"Group {group['group_id']!r} conditions must be {sorted(required)}, "
            f"got {sorted(conditions)}"
        )
    return conditions


def build_group_comparisons(
    group_lookup: dict[str, dict[str, Any]],
    metrics_by_turn_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Build cautious descriptive comparisons for the three focus groups."""
    comparisons: dict[str, Any] = {}
    for target in FOCUS_TARGETS:
        group = group_lookup[target]
        conditions = _conditions_by_target(target, group, metrics_by_turn_id)
        comparison: dict[str, Any] = {
            "group_id": group["group_id"],
            "target_variable": target,
            "conditions": conditions,
        }

        if target == "rate":
            normal_duration = conditions["normal"]["audio_duration_sec"]
            for observation in conditions.values():
                observation["duration_relative_to_normal"] = round(
                    observation["audio_duration_sec"] / normal_duration, 6
                )
            slow_duration = conditions["slow"]["audio_duration_sec"]
            fast_duration = conditions["fast"]["audio_duration_sec"]
            relative_difference = abs(slow_duration - normal_duration) / normal_duration
            weak_separation = relative_difference < 0.05
            comparison["slow_normal_relative_difference"] = round(
                relative_difference, 6
            )
            comparison["slow_normal_weak_separation_warning"] = (
                "Slow and normal durations differ by less than 5%; this is weak "
                "duration separation and requires further evaluation."
                if weak_separation
                else None
            )
            comparison["rate_direction_check"] = {
                "expected": "slow duration > normal duration > fast duration",
                "matches_expected_direction": (
                    slow_duration > normal_duration > fast_duration
                ),
                "diagnostic_only": True,
            }
            comparison["interpretation"] = (
                "Duration direction and separation are diagnostics only and do not "
                "establish rate-control fidelity."
            )
        else:
            comparison["interpretation"] = DESCRIPTIVE_ONLY_NOTE
        comparisons[target] = comparison
    return comparisons


def build_analysis(
    results: dict[str, Any], results_path: Path, generated_at: datetime
) -> dict[str, Any]:
    """Measure referenced WAVs and construct a machine-readable report."""
    turn_results = results["turn_results"]
    group_lookup = build_focus_group_lookup(turn_results)
    per_turn_metrics = []
    metrics_by_turn_id: dict[int, dict[str, Any]] = {}

    for turn in turn_results:
        turn_id = turn.get("turn_id")
        output_path = turn.get("output_path")
        if not isinstance(turn_id, int) or not isinstance(output_path, str):
            raise AcousticAnalysisError(
                "Every turn result needs an integer turn_id and string output_path"
            )
        if turn_id in metrics_by_turn_id:
            raise AcousticAnalysisError(f"Duplicate benchmark turn_id: {turn_id}")
        wav_path = resolve_wav_path(results_path, output_path)
        metrics = measure_wav(wav_path)
        metrics_by_turn_id[turn_id] = metrics
        per_turn_metrics.append(
            {
                "turn_id": turn_id,
                "group_id": turn.get("group_id"),
                "target_variable": turn.get("target_variable"),
                "target_condition": turn.get("target_condition"),
                "output_path": output_path,
                **metrics,
            }
        )

    return {
        "analysis_version": "0.1",
        "source_benchmark_results_file": results_path.name,
        "engine": results.get("engine"),
        "model": results.get("model"),
        "generated_at_utc": generated_at.isoformat(),
        "metric_definitions": {
            "audio_duration_sec": "WAV frame count divided by sample rate",
            "rms_energy": "Root mean square over all samples and channels",
            "peak_amplitude": "Maximum absolute sample amplitude",
            "silence_proportion": (
                "Proportion of non-overlapping 20 ms frames with RMS below -40 dBFS"
            ),
        },
        "pitch_analysis": {
            "status": "omitted",
            "reason": (
                "No dedicated lightweight F0 extraction dependency is declared by "
                "the project."
            ),
        },
        "per_turn_acoustic_metrics": per_turn_metrics,
        "group_comparisons": build_group_comparisons(group_lookup, metrics_by_turn_id),
        "limitations": [
            (
                "RMS, peak amplitude, silence proportion, and duration are signal "
                "descriptors; they do not establish perceptual control fidelity."
            ),
            (
                "Arousal and coarse-affect comparisons are descriptive only and "
                "require later listening or validated prosodic evaluation."
            ),
            (
                "F0 metrics are omitted because the project has no declared "
                "lightweight pitch extractor."
            ),
            (
                "Pause-before and pause-after are pipeline timing controls and are "
                "not evaluated as speech-prosody controls here."
            ),
        ],
    }


def analyze_benchmark_run(
    results_path: Path,
    *,
    generated_at: datetime | None = None,
) -> tuple[dict[str, Any], Path]:
    """Analyse one run and write JSON alongside its benchmark results."""
    results_path = results_path.resolve()
    results = _load_results(results_path)
    analysis = build_analysis(results, results_path, generated_at or datetime.now(UTC))
    output_path = results_path.parent / ANALYSIS_FILENAME
    output_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return analysis, output_path


def print_console_table(analysis: dict[str, Any]) -> None:
    """Print the three controlled model comparisons in a compact table."""
    print(
        f"{'Group':<15} {'Condition':<11} {'Duration':>9} "
        f"{'Rel.normal':>10} {'RMS':>10} {'Peak':>10} {'Silence':>9}"
    )
    print("-" * 81)
    comparisons = analysis["group_comparisons"]
    condition_order = {
        "rate": ("slow", "normal", "fast"),
        "arousal": ("low", "high"),
        "coarse_affect": ("neutral", "distressed"),
    }
    for target in FOCUS_TARGETS:
        comparison = comparisons[target]
        for condition in condition_order[target]:
            metrics = comparison["conditions"][condition]
            relative = metrics.get("duration_relative_to_normal")
            relative_text = f"{relative:.3f}" if relative is not None else "-"
            print(
                f"{comparison['group_id']:<15} {condition:<11} "
                f"{metrics['audio_duration_sec']:>9.3f} {relative_text:>10} "
                f"{metrics['rms_energy']:>10.5f} "
                f"{metrics['peak_amplitude']:>10.5f} "
                f"{metrics['silence_proportion']:>9.3f}"
            )

    rate = comparisons["rate"]
    direction = rate["rate_direction_check"]
    print(
        "Rate direction diagnostic: "
        f"{direction['matches_expected_direction']} (diagnostic only)"
    )
    if rate["slow_normal_weak_separation_warning"]:
        print(f"Warning: {rate['slow_normal_weak_separation_warning']}")
    print(DESCRIPTIVE_ONLY_NOTE)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure lightweight acoustics in a completed TTS benchmark run"
    )
    parser.add_argument(
        "--results", type=Path, required=True, help="Path to benchmark_results.json"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        analysis, output_path = analyze_benchmark_run(args.results)
    except (AcousticAnalysisError, OSError) as error:
        print(f"Acoustic analysis failed: {error}", file=sys.stderr)
        return 2
    print_console_table(analysis)
    print(f"Analysis: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
