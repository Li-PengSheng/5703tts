"""Offline tests for completed-run acoustic analysis."""

import importlib.util
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = ROOT / "scripts" / "analyze_controlled_tts_benchmark.py"
ANALYZER_SPEC = importlib.util.spec_from_file_location(
    "benchmark_acoustic_analysis", ANALYZER_PATH
)
assert ANALYZER_SPEC is not None and ANALYZER_SPEC.loader is not None
analysis_module = importlib.util.module_from_spec(ANALYZER_SPEC)
ANALYZER_SPEC.loader.exec_module(analysis_module)


def _write_wav(
    path: Path,
    duration_sec: float,
    *,
    amplitude: float = 0.25,
    silence_proportion: float = 0.0,
    sample_rate: int = 8_000,
) -> Path:
    frame_count = round(duration_sec * sample_rate)
    samples = np.full(frame_count, amplitude, dtype=np.float32)
    samples[: round(frame_count * silence_proportion)] = 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, sample_rate, subtype="FLOAT")
    return path


def _turn(
    turn_id: int,
    group_id: str,
    target_variable: str,
    target_condition: str,
    output_path: str,
) -> dict[str, Any]:
    return {
        "turn_id": turn_id,
        "group_id": group_id,
        "target_variable": target_variable,
        "target_condition": target_condition,
        "output_path": output_path,
    }


def _write_synthetic_run(
    tmp_path: Path,
    *,
    slow_duration: float = 1.2,
) -> tuple[Path, list[dict[str, Any]]]:
    run_dir = tmp_path / "cosyvoice3_test"
    durations = {
        1: slow_duration,
        2: 1.0,
        3: 0.8,
        8: 0.7,
        9: 0.7,
        10: 0.6,
        11: 0.6,
    }
    amplitudes = {1: 0.2, 2: 0.2, 3: 0.2, 8: 0.1, 9: 0.3, 10: 0.15, 11: 0.25}
    turns = [
        _turn(1, "rate", "rate", "slow", "turns/turn_001.wav"),
        _turn(2, "rate", "rate", "normal", "turns/turn_002.wav"),
        _turn(3, "rate", "rate", "fast", "turns/turn_003.wav"),
        _turn(8, "arousal", "arousal", "low", "turns/turn_008.wav"),
        _turn(9, "arousal", "arousal", "high", "turns/turn_009.wav"),
        _turn(
            10,
            "coarse_affect",
            "coarse_affect",
            "neutral",
            "turns/turn_010.wav",
        ),
        _turn(
            11,
            "coarse_affect",
            "coarse_affect",
            "distressed",
            "turns/turn_011.wav",
        ),
    ]
    for turn in turns:
        _write_wav(
            run_dir / turn["output_path"],
            durations[turn["turn_id"]],
            amplitude=amplitudes[turn["turn_id"]],
        )

    results_path = run_dir / "benchmark_results.json"
    results_path.write_text(
        json.dumps(
            {
                "engine": "cosyvoice",
                "model": "synthetic-test-model",
                "turn_results": turns,
            }
        ),
        encoding="utf-8",
    )
    return results_path, turns


def test_measure_wav_duration_rms_peak_and_silence(tmp_path: Path) -> None:
    wav_path = _write_wav(
        tmp_path / "half_silence.wav",
        1.0,
        amplitude=0.5,
        silence_proportion=0.5,
    )

    metrics = analysis_module.measure_wav(wav_path)

    assert metrics["audio_duration_sec"] == pytest.approx(1.0)
    assert metrics["rms_energy"] == pytest.approx(math.sqrt(0.125), abs=1e-8)
    assert metrics["peak_amplitude"] == pytest.approx(0.5)
    assert metrics["silence_proportion"] == pytest.approx(0.5)
    assert metrics["sample_rate"] == 8_000
    assert metrics["f0_metrics"] is None


def test_group_lookup_finds_rate_arousal_and_affect(tmp_path: Path) -> None:
    _, turns = _write_synthetic_run(tmp_path)

    lookup = analysis_module.build_focus_group_lookup(turns)

    assert set(lookup) == {"rate", "arousal", "coarse_affect"}
    assert [turn["turn_id"] for turn in lookup["rate"]["turn_results"]] == [
        1,
        2,
        3,
    ]


def test_rate_relative_duration_and_directional_diagnostic(tmp_path: Path) -> None:
    results_path, _ = _write_synthetic_run(tmp_path)

    report, _ = analysis_module.analyze_benchmark_run(
        results_path, generated_at=datetime(2026, 8, 28, tzinfo=UTC)
    )
    rate = report["group_comparisons"]["rate"]

    assert rate["conditions"]["slow"]["duration_relative_to_normal"] == 1.2
    assert rate["conditions"]["normal"]["duration_relative_to_normal"] == 1.0
    assert rate["conditions"]["fast"]["duration_relative_to_normal"] == 0.8
    assert rate["rate_direction_check"] == {
        "expected": "slow duration > normal duration > fast duration",
        "matches_expected_direction": True,
        "diagnostic_only": True,
    }
    assert rate["slow_normal_weak_separation_warning"] is None


def test_slow_normal_difference_below_five_percent_warns(tmp_path: Path) -> None:
    results_path, _ = _write_synthetic_run(tmp_path, slow_duration=1.04)

    report, _ = analysis_module.analyze_benchmark_run(results_path)
    rate = report["group_comparisons"]["rate"]

    assert rate["slow_normal_relative_difference"] == pytest.approx(0.04)
    assert "less than 5%" in rate["slow_normal_weak_separation_warning"]
    assert rate["rate_direction_check"]["matches_expected_direction"] is True


def test_analysis_json_is_written_alongside_source_results(tmp_path: Path) -> None:
    results_path, turns = _write_synthetic_run(tmp_path)
    generated_at = datetime(2026, 8, 28, 1, 2, 3, tzinfo=UTC)

    report, output_path = analysis_module.analyze_benchmark_run(
        results_path, generated_at=generated_at
    )
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path == results_path.parent / "benchmark_acoustic_analysis.json"
    assert written == report
    assert report["source_benchmark_results_file"] == "benchmark_results.json"
    assert report["engine"] == "cosyvoice"
    assert report["model"] == "synthetic-test-model"
    assert report["generated_at_utc"] == generated_at.isoformat()
    assert len(report["per_turn_acoustic_metrics"]) == len(turns)
    assert report["pitch_analysis"]["status"] == "omitted"


def test_arousal_and_affect_are_descriptive_without_scores(tmp_path: Path) -> None:
    results_path, _ = _write_synthetic_run(tmp_path)

    report, _ = analysis_module.analyze_benchmark_run(results_path)
    comparisons = report["group_comparisons"]

    assert set(comparisons["arousal"]["conditions"]) == {"low", "high"}
    assert set(comparisons["coarse_affect"]["conditions"]) == {
        "neutral",
        "distressed",
    }
    assert "requires later validation" in comparisons["arousal"]["interpretation"]
    assert "requires later validation" in comparisons["coarse_affect"]["interpretation"]
    assert "fidelity_score" not in json.dumps(comparisons)


def test_missing_referenced_wav_is_rejected(tmp_path: Path) -> None:
    results_path, _ = _write_synthetic_run(tmp_path)
    missing_path = results_path.parent / "turns" / "turn_009.wav"
    missing_path.unlink()

    with pytest.raises(
        analysis_module.AcousticAnalysisError,
        match=r"Referenced WAV file not found: .*turn_009\.wav",
    ):
        analysis_module.analyze_benchmark_run(results_path)


def test_zero_duration_wav_is_rejected(tmp_path: Path) -> None:
    empty_wav = tmp_path / "empty.wav"
    sf.write(empty_wav, np.array([], dtype=np.float32), 8_000)

    with pytest.raises(
        analysis_module.AcousticAnalysisError, match="WAV has invalid duration"
    ):
        analysis_module.measure_wav(empty_wav)


def test_console_table_contains_focus_groups(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results_path, _ = _write_synthetic_run(tmp_path)
    report, _ = analysis_module.analyze_benchmark_run(results_path)

    analysis_module.print_console_table(report)

    output = capsys.readouterr().out
    assert "rate" in output
    assert "arousal" in output
    assert "coarse_affect" in output
    assert "diagnostic only" in output
