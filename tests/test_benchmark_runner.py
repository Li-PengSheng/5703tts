"""Offline deterministic tests for the controlled benchmark runner."""

import copy
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from tts5703.config import load_config
from tts5703.validate import load_and_validate

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_controlled_tts_benchmark.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "controlled_benchmark", RUNNER_PATH
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
benchmark = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(benchmark)
FIXTURE_PATH = ROOT / "data" / "benchmark" / "benchmark_fixture_v0.1.json"
MANIFEST_PATH = ROOT / "data" / "benchmark" / "benchmark_manifest_v0.1.json"
CONFIG_PATH = ROOT / "config" / "config.yaml"
KOKORO_CONFIG_PATH = ROOT / "config" / "config.kokoro.yaml"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _non_rate_design() -> tuple[dict[str, Any], dict[str, Any]]:
    fixture = copy.deepcopy(_load(FIXTURE_PATH))
    fixture["dialogue_id"] = "non_rate_benchmark"
    fixture["turns"] = [
        turn for turn in fixture["turns"] if turn["turn_id"] in {4, 5, 8, 9}
    ]
    manifest = copy.deepcopy(_load(MANIFEST_PATH))
    manifest["fixture"] = "non_rate_fixture.json"
    manifest["groups"] = [manifest["groups"][1], manifest["groups"][3]]
    return fixture, manifest


class StepClock:
    def __init__(self, step: float = 0.5) -> None:
        self.value = -step
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


def _fixed_utc_now() -> datetime:
    return datetime(2026, 8, 28, 12, 0, tzinfo=UTC)


def _install_fake_synthesizer(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[int],
    *,
    fail_turn_id: int | None = None,
    zero_duration_turn_id: int | None = None,
    unexpected_rate_direction: bool = False,
) -> None:
    async def fake_synthesize_turn(turn, out_dir: Path, config: dict) -> Path:
        calls.append(turn.turn_id)
        if turn.turn_id == fail_turn_id:
            raise RuntimeError("fake synthesis failure")
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"turn_{turn.turn_id:03d}.wav"
        if turn.turn_id == zero_duration_turn_id:
            samples = np.array([], dtype=np.float32)
        else:
            rate_durations = (
                {"slow": 0.1, "normal": 0.2, "fast": 0.3}
                if unexpected_rate_direction
                else {"slow": 0.3, "normal": 0.2, "fast": 0.1}
            )
            duration = rate_durations.get(turn.rate, 0.2)
            samples = np.zeros(round(8_000 * duration), dtype=np.float32)
        sf.write(output, samples, 8_000)
        return output

    monkeypatch.setattr(benchmark, "synthesize_turn", fake_synthesize_turn)


def _run_fake_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    warmup_enabled: bool = True,
    fail_turn_id: int | None = None,
    zero_duration_turn_id: int | None = None,
    unexpected_rate_direction: bool = False,
    config_path: Path = CONFIG_PATH,
    run_id: str | None = "cosyvoice3_test",
) -> tuple[dict[str, Any], Path, list[int]]:
    calls: list[int] = []
    _install_fake_synthesizer(
        monkeypatch,
        calls,
        fail_turn_id=fail_turn_id,
        zero_duration_turn_id=zero_duration_turn_id,
        unexpected_rate_direction=unexpected_rate_direction,
    )
    results, run_dir = benchmark.run_benchmark(
        config_path=config_path,
        fixture_path=FIXTURE_PATH,
        manifest_path=MANIFEST_PATH,
        output_root=tmp_path / "runs",
        warmup_enabled=warmup_enabled,
        run_id=run_id,
        clock=StepClock(),
        utc_now=_fixed_utc_now,
    )
    return results, run_dir, calls


def test_cosyvoice_and_kokoro_are_supported_benchmark_engines() -> None:
    assert benchmark.benchmark_engine(load_config(CONFIG_PATH)) == "cosyvoice"
    assert benchmark.benchmark_engine(load_config(KOKORO_CONFIG_PATH)) == "kokoro"


def test_other_production_engine_is_rejected_for_controlled_benchmark() -> None:
    with pytest.raises(
        benchmark.BenchmarkDesignError,
        match="Unsupported controlled benchmark engine: 'edge_tts'",
    ):
        benchmark.benchmark_engine({"tts": {"engine": "edge_tts"}})


def test_unsupported_engine_is_rejected_before_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    _install_fake_synthesizer(monkeypatch, calls)
    monkeypatch.setattr(
        benchmark, "load_config", lambda _path: {"tts": {"engine": "edge_tts"}}
    )

    with pytest.raises(
        benchmark.BenchmarkDesignError,
        match="Unsupported controlled benchmark engine: 'edge_tts'",
    ):
        benchmark.run_benchmark(
            config_path=tmp_path / "unused.yaml",
            output_root=tmp_path / "runs",
        )

    assert calls == []
    assert not (tmp_path / "runs").exists()


def test_engine_capabilities_distinguish_model_pipeline_and_unsupported() -> None:
    cosyvoice = benchmark.engine_capabilities("cosyvoice")
    kokoro = benchmark.engine_capabilities("kokoro")

    assert cosyvoice["rate"]["support"] == "model_control"
    assert cosyvoice["arousal"]["support"] == "provisional_model_control"
    assert cosyvoice["coarse_affect"]["support"] == ("provisional_model_control")
    assert kokoro["rate"]["support"] == "model_control"
    assert kokoro["pause_before_ms"]["support"] == "pipeline_timing"
    assert kokoro["pause_after_ms"]["support"] == "pipeline_timing"
    assert kokoro["arousal"]["support"] == "unsupported"
    assert kokoro["coarse_affect"]["support"] == "unsupported"


def test_manifest_lookup_maps_every_fixture_turn_exactly_once() -> None:
    fixture = _load(FIXTURE_PATH)
    lookup = benchmark.build_manifest_lookup(fixture, _load(MANIFEST_PATH))

    assert set(lookup) == {turn["turn_id"] for turn in fixture["turns"]}
    assert lookup[1] == {
        "group_id": "rate",
        "target_variable": "rate",
        "target_condition": "slow",
    }


@pytest.mark.parametrize("group_id", [None, "", "   "])
def test_group_id_must_be_present_and_non_empty(group_id: str | None) -> None:
    manifest = copy.deepcopy(_load(MANIFEST_PATH))
    manifest["groups"][0]["group_id"] = group_id

    with pytest.raises(benchmark.BenchmarkDesignError, match="non-empty.*group_id"):
        benchmark.build_manifest_lookup(_load(FIXTURE_PATH), manifest)


def test_duplicate_group_ids_are_rejected_before_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = copy.deepcopy(_load(MANIFEST_PATH))
    manifest["groups"][1]["group_id"] = manifest["groups"][0]["group_id"]
    manifest_path = _write_json(tmp_path / "duplicate_groups.json", manifest)
    calls: list[int] = []
    _install_fake_synthesizer(monkeypatch, calls)

    with pytest.raises(
        benchmark.BenchmarkDesignError,
        match=r"Duplicate benchmark group_id: 'rate'",
    ):
        benchmark.run_benchmark(
            config_path=CONFIG_PATH,
            fixture_path=FIXTURE_PATH,
            manifest_path=manifest_path,
            output_root=tmp_path / "runs",
            warmup_enabled=False,
        )

    assert calls == []
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("text", "This deliberately changes the controlled comparison text."),
        ("speaker", "counsellor"),
        ("label", "normal"),
    ],
)
def test_group_top_level_confound_is_rejected(field: str, replacement: str) -> None:
    fixture = copy.deepcopy(_load(FIXTURE_PATH))
    fixture["turns"][1][field] = replacement

    with pytest.raises(
        benchmark.BenchmarkDesignError,
        match=rf"Group 'rate' is confounded: {field} differs between turns 1 and 2",
    ):
        benchmark.build_manifest_lookup(fixture, _load(MANIFEST_PATH))


def test_non_target_arousal_confound_in_rate_group_is_rejected() -> None:
    fixture = copy.deepcopy(_load(FIXTURE_PATH))
    fixture["turns"][1]["acoustic_spec"]["arousal"] = "high"

    with pytest.raises(
        benchmark.BenchmarkDesignError,
        match=(
            r"Group 'rate' is confounded: non-target acoustic field 'arousal' "
            r"differs between turns 1 and 2"
        ),
    ):
        benchmark.build_manifest_lookup(fixture, _load(MANIFEST_PATH))


def test_non_target_affect_confound_in_arousal_group_is_rejected() -> None:
    fixture = copy.deepcopy(_load(FIXTURE_PATH))
    fixture["turns"][8]["acoustic_spec"]["coarse_affect"] = "neutral"

    with pytest.raises(
        benchmark.BenchmarkDesignError,
        match=(
            r"Group 'arousal' is confounded: non-target acoustic field "
            r"'coarse_affect' differs between turns 8 and 9"
        ),
    ):
        benchmark.build_manifest_lookup(fixture, _load(MANIFEST_PATH))


def test_target_variable_must_actually_vary() -> None:
    fixture = copy.deepcopy(_load(FIXTURE_PATH))
    manifest = copy.deepcopy(_load(MANIFEST_PATH))
    for turn in fixture["turns"][:3]:
        turn["acoustic_spec"]["rate"] = "normal"
    manifest["groups"][0]["expected_conditions"] = ["normal"] * 3

    with pytest.raises(
        benchmark.BenchmarkDesignError,
        match=r"Group 'rate' target acoustic field 'rate' does not vary",
    ):
        benchmark.build_manifest_lookup(fixture, manifest)


def test_manifest_condition_mismatch_is_rejected_before_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _load(MANIFEST_PATH)
    manifest["groups"][0]["expected_conditions"][0] = "normal"
    manifest_path = tmp_path / "bad_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    calls: list[int] = []
    _install_fake_synthesizer(monkeypatch, calls)

    with pytest.raises(benchmark.BenchmarkDesignError, match="does not match"):
        benchmark.run_benchmark(
            config_path=CONFIG_PATH,
            fixture_path=FIXTURE_PATH,
            manifest_path=manifest_path,
            output_root=tmp_path / "runs",
        )

    assert calls == []
    assert not (tmp_path / "runs").exists()


def test_confound_is_rejected_before_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = copy.deepcopy(_load(FIXTURE_PATH))
    fixture["turns"][1]["text"] = "This text creates an invalid comparison."
    fixture_path = tmp_path / "confounded_fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    calls: list[int] = []
    _install_fake_synthesizer(monkeypatch, calls)

    with pytest.raises(benchmark.BenchmarkDesignError, match="text differs"):
        benchmark.run_benchmark(
            config_path=CONFIG_PATH,
            fixture_path=fixture_path,
            manifest_path=MANIFEST_PATH,
            output_root=tmp_path / "runs",
        )

    assert calls == []
    assert not (tmp_path / "runs").exists()


def test_duplicate_manifest_membership_is_rejected() -> None:
    fixture = _load(FIXTURE_PATH)
    manifest = _load(MANIFEST_PATH)
    manifest["groups"][1]["turn_ids"].append(1)
    manifest["groups"][1]["expected_conditions"].append(0)

    with pytest.raises(benchmark.BenchmarkDesignError, match="more than one"):
        benchmark.build_manifest_lookup(fixture, manifest)


def test_missing_manifest_membership_is_rejected() -> None:
    fixture = _load(FIXTURE_PATH)
    manifest = _load(MANIFEST_PATH)
    manifest["groups"][-1]["turn_ids"].pop()
    manifest["groups"][-1]["expected_conditions"].pop()

    with pytest.raises(benchmark.BenchmarkDesignError, match="missing"):
        benchmark.build_manifest_lookup(fixture, manifest)


def test_effective_trace_uses_existing_cosyvoice_mapping() -> None:
    config = load_config(CONFIG_PATH)
    dialogue = load_and_validate(FIXTURE_PATH, config)
    high_arousal_turn = next(turn for turn in dialogue.turns if turn.turn_id == 9)

    controls = benchmark.effective_cosyvoice_controls(
        high_arousal_turn, config, Path("turn_009.wav")
    )

    assert controls["speed"] == 1.0
    assert controls["mode"] == "instruct2"
    assert "energetic, intense" in controls["instruction"]
    assert "distressed, worried, and sad" in controls["instruction"]


@pytest.mark.parametrize(
    ("turn_id", "expected_speed"),
    [(1, 0.8), (2, 1.0), (3, 1.2)],
)
def test_kokoro_effective_trace_reuses_production_rate_mapping(
    turn_id: int, expected_speed: float
) -> None:
    config = load_config(KOKORO_CONFIG_PATH)
    dialogue = load_and_validate(FIXTURE_PATH, config)
    turn = next(turn for turn in dialogue.turns if turn.turn_id == turn_id)

    controls = benchmark.effective_controls_for_turn(
        "kokoro", turn, config, Path(f"turn_{turn_id:03d}.wav")
    )

    assert controls["speed"] == expected_speed
    assert controls["voice"] == "am_adam"


def test_kokoro_trace_records_configured_voice_and_ignored_requested_controls() -> None:
    config = load_config(KOKORO_CONFIG_PATH)
    dialogue = load_and_validate(FIXTURE_PATH, config)
    high_arousal_turn = next(turn for turn in dialogue.turns if turn.turn_id == 9)
    counsellor_turn = next(turn for turn in dialogue.turns if turn.turn_id == 10)

    high_controls = benchmark.effective_controls_for_turn(
        "kokoro", high_arousal_turn, config, Path("turn_009.wav")
    )
    counsellor_controls = benchmark.effective_controls_for_turn(
        "kokoro", counsellor_turn, config, Path("turn_010.wav")
    )

    assert high_controls == {
        "speed": 1.0,
        "voice": "am_adam",
        "ignored_requested_controls": ["arousal", "coarse_affect"],
    }
    assert counsellor_controls["voice"] == "af_heart"
    assert counsellor_controls["ignored_requested_controls"] == [
        "arousal",
        "coarse_affect",
    ]


def test_rtf_calculation() -> None:
    assert benchmark.calculate_rtf(2.0, 4.0) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="greater than zero"):
        benchmark.calculate_rtf(1.0, 0.0)


def test_successful_fake_run_records_trace_timing_outputs_and_pause_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, run_dir, calls = _run_fake_benchmark(tmp_path, monkeypatch)

    assert results["status"] == "complete"
    assert calls == [2, *range(1, 12)]
    assert results["warmup"]["enabled"] is True
    assert results["warmup"]["turn_id"] == 2
    assert results["warmup"]["elapsed_sec"] == 0.5
    assert len(results["turn_results"]) == 11
    first = results["turn_results"][0]
    assert first["generation_elapsed_sec"] == 0.5
    assert first["audio_duration_sec"] == pytest.approx(0.3)
    assert first["real_time_factor"] == pytest.approx(0.5 / 0.3, abs=1e-6)
    assert first["sample_rate"] == 8_000
    assert first["output_path"] == "turns/turn_001.wav"
    assert first["requested_acoustic_spec"]["rate"] == "slow"
    assert first["requested_acoustic_spec"]["pause_after_ms"] == 500
    assert first["target_control_support"] == "model_control"
    assert first["effective_controls"]["speed"] == 0.8
    assert first["effective_controls"]["mode"] == "instruct2"
    assert first["effective_controls"]["instruction"] is not None
    assert first["worker_elapsed_sec"] is None
    assert results["turn_results"][8]["target_control_support"] == (
        "provisional_model_control"
    )
    assert (run_dir / first["output_path"]).is_file()
    assert (run_dir / "warmup" / "turn_002.wav").is_file()
    assert (run_dir / "benchmark_clean.wav").is_file()
    assert (run_dir / benchmark.RESULTS_FILENAME).is_file()
    assert results["assembly"]["pause_diagnostic"]["passed"] is True
    assert results["assembly"]["pause_diagnostic"]["control_type"] == (
        "pipeline_timing"
    )
    assert results["control_scope"] == {
        "model_controls": ["rate", "arousal", "coarse_affect"],
        "pipeline_timing_controls": ["pause_before_ms", "pause_after_ms"],
        "unsupported_controls": ["emotion", "paralinguistic_events"],
    }
    assert results["control_capabilities"]["arousal"]["support"] == (
        "provisional_model_control"
    )
    assert results["group_summaries"]["rate"]["rate_direction_check"] == {
        "expected": "slow duration > normal duration > fast duration",
        "matches_expected_direction": True,
        "diagnostic_only": True,
    }
    assert "fidelity_score" not in results["group_summaries"]["arousal"]
    assert "fidelity_score" not in results["group_summaries"]["coarse_affect"]


def test_successful_kokoro_fake_run_records_capabilities_and_unsupported_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, run_dir, calls = _run_fake_benchmark(
        tmp_path,
        monkeypatch,
        config_path=KOKORO_CONFIG_PATH,
        run_id=None,
    )

    assert results["status"] == "complete"
    assert calls == [2, *range(1, 12)]
    assert run_dir.name == "kokoro_20260828T120000Z"
    assert results["engine"] == "kokoro"
    assert results["model"] == "Kokoro-82M"
    assert results["environment"] == {
        **{
            key: results["environment"][key]
            for key in ("platform", "python_version", "git_commit")
        },
        "model": "Kokoro-82M",
        "lang_code": "a",
        "sample_rate": 24_000,
        "device": None,
        "voice_map": {"counsellor": "af_heart", "caller": "am_adam"},
    }
    assert results["control_scope"] == {
        "model_controls": ["rate"],
        "pipeline_timing_controls": ["pause_before_ms", "pause_after_ms"],
        "unsupported_controls": [
            "arousal",
            "coarse_affect",
            "emotion",
            "paralinguistic_events",
        ],
    }

    turns_by_id = {turn["turn_id"]: turn for turn in results["turn_results"]}
    assert turns_by_id[1]["target_control_support"] == "model_control"
    assert turns_by_id[4]["target_control_support"] == "pipeline_timing"
    assert turns_by_id[6]["target_control_support"] == "pipeline_timing"
    assert turns_by_id[8]["target_control_support"] == "unsupported"
    assert turns_by_id[10]["target_control_support"] == "unsupported"
    assert turns_by_id[1]["effective_controls"] == {
        "speed": 0.8,
        "voice": "am_adam",
        "ignored_requested_controls": ["arousal", "coarse_affect"],
    }

    arousal = results["group_summaries"]["arousal"]
    affect = results["group_summaries"]["coarse_affect"]
    assert results["group_summaries"]["pause_before"]["support_type"] == (
        "pipeline_timing"
    )
    assert results["group_summaries"]["pause_after"]["control_supported"] is True
    for summary in (arousal, affect):
        assert summary["control_supported"] is False
        assert summary["support_type"] == "unsupported"
        assert summary["requested_target_control_ignored"] is True
        assert summary["controllability_conclusion"] is None
        assert "not evidence of controllability" in summary["interpretation"]
        assert all(
            observation["generation_success"] is True
            for observation in summary["observations"]
        )
        assert "control_success" not in summary
    assert (
        results["group_summaries"]["rate"]["rate_direction_check"][
            "matches_expected_direction"
        ]
        is True
    )
    assert results["assembly"]["pause_diagnostic"]["passed"] is True


def test_kokoro_no_warmup_skips_baseline_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, _, calls = _run_fake_benchmark(
        tmp_path,
        monkeypatch,
        config_path=KOKORO_CONFIG_PATH,
        warmup_enabled=False,
        run_id="kokoro_no_warmup",
    )

    assert calls == list(range(1, 12))
    assert results["warmup"] == {
        "enabled": False,
        "turn_id": None,
        "elapsed_sec": None,
    }


def test_no_warmup_skips_extra_synthesis(tmp_path: Path, monkeypatch) -> None:
    results, _, calls = _run_fake_benchmark(tmp_path, monkeypatch, warmup_enabled=False)

    assert calls == list(range(1, 12))
    assert len(results["turn_results"]) == 11
    assert results["warmup"] == {
        "enabled": False,
        "turn_id": None,
        "elapsed_sec": None,
    }


def test_no_warmup_allows_valid_manifest_without_rate_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, manifest = _non_rate_design()
    fixture_path = _write_json(tmp_path / "non_rate_fixture.json", fixture)
    manifest_path = _write_json(tmp_path / "non_rate_manifest.json", manifest)
    calls: list[int] = []
    _install_fake_synthesizer(monkeypatch, calls)

    results, _ = benchmark.run_benchmark(
        config_path=CONFIG_PATH,
        fixture_path=fixture_path,
        manifest_path=manifest_path,
        output_root=tmp_path / "runs",
        warmup_enabled=False,
        run_id="non_rate_test",
        clock=StepClock(),
        utc_now=_fixed_utc_now,
    )

    assert results["status"] == "complete"
    assert calls == [4, 5, 8, 9]
    assert results["warmup"] == {
        "enabled": False,
        "turn_id": None,
        "elapsed_sec": None,
    }
    assert set(results["group_summaries"]) == {"pause_before", "arousal"}
    assert len(results["group_summaries"]) == len(manifest["groups"])


def test_warmup_enabled_still_requires_rate_baseline_for_custom_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, manifest = _non_rate_design()
    fixture_path = _write_json(tmp_path / "non_rate_fixture.json", fixture)
    manifest_path = _write_json(tmp_path / "non_rate_manifest.json", manifest)
    calls: list[int] = []
    _install_fake_synthesizer(monkeypatch, calls)

    with pytest.raises(
        benchmark.BenchmarkDesignError,
        match="exactly one rate group",
    ):
        benchmark.run_benchmark(
            config_path=CONFIG_PATH,
            fixture_path=fixture_path,
            manifest_path=manifest_path,
            output_root=tmp_path / "runs",
            warmup_enabled=True,
        )

    assert calls == []
    assert not (tmp_path / "runs").exists()


def test_invalid_wav_duration_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, run_dir, calls = _run_fake_benchmark(
        tmp_path,
        monkeypatch,
        warmup_enabled=False,
        zero_duration_turn_id=1,
    )

    assert calls == [1]
    assert results["status"] == "failed"
    assert results["failure"]["turn_id"] == 1
    assert "invalid duration" in results["failure"]["message"]
    assert results["turn_results"] == []
    persisted = _load(run_dir / benchmark.RESULTS_FILENAME)
    assert persisted["status"] == "failed"


def test_failed_turn_preserves_partial_results_and_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, run_dir, calls = _run_fake_benchmark(
        tmp_path, monkeypatch, warmup_enabled=False, fail_turn_id=3
    )

    assert calls == [1, 2, 3]
    assert results["status"] == "failed"
    assert results["failure"]["turn_id"] == 3
    assert [result["turn_id"] for result in results["turn_results"]] == [1, 2]
    assert (run_dir / "turns" / "turn_001.wav").is_file()
    assert (run_dir / "turns" / "turn_002.wav").is_file()
    assert not (run_dir / "turns" / "turn_003.wav").exists()
    assert _load(run_dir / benchmark.RESULTS_FILENAME)["status"] == "failed"


def test_unexpected_rate_direction_is_diagnostic_not_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, _, _ = _run_fake_benchmark(
        tmp_path,
        monkeypatch,
        warmup_enabled=False,
        unexpected_rate_direction=True,
    )

    assert results["status"] == "complete"
    diagnostic = results["group_summaries"]["rate"]["rate_direction_check"]
    assert diagnostic["matches_expected_direction"] is False
    assert diagnostic["diagnostic_only"] is True


def test_default_run_directory_is_gitignored() -> None:
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "-q",
            "data/benchmark/runs/example/turns/turn_001.wav",
        ],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0


def test_runner_import_does_not_load_model_packages() -> None:
    assert not any(
        name == "cosyvoice" or name.startswith("cosyvoice.") for name in sys.modules
    )
    assert not any(
        name == "kokoro" or name.startswith("kokoro.") for name in sys.modules
    )
