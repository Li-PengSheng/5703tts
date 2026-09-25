"""Frozen Higgs signal policy and durable quality-evidence checks."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tts5703 import audio_quality as quality


def wav(
    path: Path, sections: list[tuple[float, float]], *, rate=8000, channels=1
) -> None:
    samples = np.concatenate(
        [
            np.full((round(seconds * rate), channels), amplitude, dtype=np.float32)
            for seconds, amplitude in sections
        ]
    )
    sf.write(path, samples, rate, subtype="FLOAT")


def evidence(tmp_path: Path, *, amplitude=0.5) -> dict:
    wav(tmp_path / "turn_001.wav", [(2, amplitude)])
    turns = [
        {
            "source_identity": {"ordinal": 1, "source_turn_id": "source-1"},
            "execution": {"turn_audio": "turn_001.wav"},
        }
    ]
    return quality.assess_dialogue_turns(
        tmp_path, "D", "render-fingerprint", turns, assessment="new_render"
    )


def test_frozen_policy_identity_and_signal_vector(tmp_path: Path) -> None:
    assert quality.policy_identity()["sha256"] == (
        "134cc23da9efd8b1f7e3445232033c369675780512aa3b1707908b2f8adb7b52"
    )
    path = tmp_path / "vector.wav"
    wav(path, [(5, 0.5), (7, 0.02)], rate=16000, channels=2)
    metrics = quality.audio_metrics(path)
    assert metrics["duration_sec"] == 12
    assert metrics["body_dbfs"] == pytest.approx(10 * math.log10(0.25))
    assert metrics["tail_5_drop_db"] == pytest.approx(20 * math.log10(25), abs=1e-5)
    assert metrics["relative_thresholds"]["10"]["trailing_low_sec"] == 7
    assert metrics["tail_dbfs"]["5"] > -40
    assert quality.assess_turn(path)["issues"][0]["code"] == "abnormal_tail"


@pytest.mark.parametrize("rate,channels", [(8000, 1), (16000, 2), (44100, 1)])
def test_partial_block_and_multiple_rates(
    tmp_path: Path, rate: int, channels: int
) -> None:
    path = tmp_path / "partial.wav"
    wav(path, [(5, 0.5), (0.1, 0.02)], rate=rate, channels=channels)
    metrics = quality.audio_metrics(path)
    assert metrics["duration_sec"] == pytest.approx(5.1, abs=1 / rate)
    assert metrics["relative_thresholds"]["10"]["trailing_low_sec"] == pytest.approx(
        0.1, abs=1 / rate
    )
    assert metrics["relative_thresholds"]["10"][
        "low_fraction_last_5_sec"
    ] == pytest.approx(0.02, abs=1 / rate)
    assert metrics["tail_dbfs"]["5"] == pytest.approx(
        10 * math.log10((4.9 * 0.25 + 0.1 * 0.02**2) / 5), abs=0.01
    )


def test_linear_body_median_and_digital_zero(tmp_path: Path) -> None:
    path = tmp_path / "mixed.wav"
    wav(path, [(0.25, 0), (1.75, 0.5)])
    assert quality.audio_metrics(path)["body_dbfs"] == pytest.approx(
        10 * math.log10(0.125)
    )
    wav(path, [(0.5, 0), (2.5, 0.5)])
    assert quality.audio_metrics(path)["body_dbfs"] == -math.inf
    assert quality.assess_turn(path)["outcome"] == "indeterminate"
    wav(path, [(2, 0)])
    all_zero = quality.assess_turn(path)
    assert all_zero["outcome"] == "reject"
    assert all_zero["issues"][0]["code"] == "exact_digital_silence"


def test_finite_body_and_exact_zero_tail_is_abnormal_tail(tmp_path: Path) -> None:
    path = tmp_path / "zero_tail.wav"
    wav(path, [(5, 0.5), (5, 0)])
    metrics = quality.audio_metrics(path)
    assert metrics["body_dbfs"] == pytest.approx(10 * math.log10(0.25))
    assert metrics["tail_5_drop_db"] == math.inf
    assert metrics["relative_thresholds"]["10"]["trailing_low_sec"] == 5
    assessed = quality.assess_turn(path)
    assert assessed["outcome"] == "reject"
    assert assessed["issues"][0]["code"] == "abnormal_tail"


@pytest.mark.parametrize("drop,trailing", [(None, 3), (math.nan, 3), (10, math.nan)])
def test_unavailable_or_nan_metric_is_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drop: float | None,
    trailing: float,
) -> None:
    monkeypatch.setattr(
        quality,
        "audio_metrics",
        lambda _: {
            "exact_digital_zero": False,
            "tail_5_drop_db": drop,
            "relative_thresholds": {"10": {"trailing_low_sec": trailing}},
        },
    )
    assessed = quality.assess_turn(tmp_path / "unused.wav")
    assert assessed["outcome"] == "indeterminate"
    assert assessed["issues"][0]["code"] == "metrics_unavailable"


def test_nan_metric_still_commits_indeterminate_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav(tmp_path / "turn_001.wav", [(2, 0.5)])
    monkeypatch.setattr(
        quality,
        "audio_metrics",
        lambda _: {
            "exact_digital_zero": False,
            "tail_5_drop_db": math.nan,
            "relative_thresholds": {"10": {"trailing_low_sec": 3.0}},
        },
    )
    turns = [
        {
            "source_identity": {"ordinal": 1, "source_turn_id": "source-1"},
            "execution": {"turn_audio": "turn_001.wav"},
        }
    ]
    item = quality.assess_dialogue_turns(
        tmp_path, "D", "render-fingerprint", turns, assessment="new_render"
    )
    assert item["outcome"] == "indeterminate"
    assert item["issues"][0]["code"] == "metrics_unavailable"
    quality.write_quality_sidecar_atomic(tmp_path, "D", item, ["turn_001.wav"])


@pytest.mark.parametrize(
    "drop,trailing,outcome",
    [
        (10, 3, "reject"),
        (9.999, 3, "pass"),
        (10, 2.999, "pass"),
        (math.inf, 3, "reject"),
        (math.inf, 2.999, "pass"),
    ],
)
def test_inclusive_and_predicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drop: float,
    trailing: float,
    outcome: str,
) -> None:
    monkeypatch.setattr(
        quality,
        "audio_metrics",
        lambda _: {
            "exact_digital_zero": False,
            "tail_5_drop_db": drop,
            "relative_thresholds": {"10": {"trailing_low_sec": trailing}},
        },
    )
    assert quality.assess_turn(tmp_path / "unused.wav")["outcome"] == outcome


def test_decoder_error_is_indeterminate(tmp_path: Path) -> None:
    path = tmp_path / "broken.wav"
    path.write_bytes(b"not a WAV")
    assert quality.assess_turn(path)["issues"][0]["code"] == "metrics_unavailable"


def test_sidecar_roundtrip_complete_hashes_and_validation(tmp_path: Path) -> None:
    item = evidence(tmp_path)
    path = quality.write_quality_sidecar_atomic(tmp_path, "D", item, ["turn_001.wav"])
    assert quality.load_quality_sidecar(tmp_path, "D", ["turn_001.wav"]) == item
    assert quality.verify_turn_hashes(tmp_path, item) is None
    wav(tmp_path / "turn_001.wav", [(2, 0.1)])
    assert "hash changed" in quality.verify_turn_hashes(tmp_path, item)
    assert json.loads(path.read_text())["policy"]["id"] == quality.POLICY_ID
    for changed in (
        {**item, "turn_wav_sha256": {}},
        {**item, "render_fingerprint": None},
        {**item, "policy": {**item["policy"], "sha256": "0" * 64}},
        {**item, "schema_version": "2.0"},
    ):
        with pytest.raises(quality.QualityEvidenceError):
            quality.validate_quality_sidecar(changed, "D", ["turn_001.wav"])


def test_rejected_turn_still_hashes_passing_sibling(tmp_path: Path) -> None:
    wav(tmp_path / "turn_001.wav", [(5, 0.5), (7, 0.02)])
    wav(tmp_path / "turn_002.wav", [(2, 0.5)])
    turns = [
        {
            "source_identity": {
                "ordinal": ordinal,
                "source_turn_id": f"source-{ordinal}",
            },
            "execution": {"turn_audio": f"turn_{ordinal:03d}.wav"},
        }
        for ordinal in (1, 2)
    ]
    item = quality.assess_dialogue_turns(
        tmp_path, "D", "render-fingerprint", turns, assessment="new_render"
    )
    assert item["outcome"] == "reject"
    assert set(item["turn_wav_sha256"]) == {"turn_001.wav", "turn_002.wav"}
    assert item["issues"][0]["ordinal"] == 1


@pytest.mark.parametrize("unsafe_kind", ["symlink", "directory"])
def test_unsafe_existing_sidecar_blocks_write(tmp_path: Path, unsafe_kind: str) -> None:
    item = evidence(tmp_path)
    path = quality.quality_sidecar_path(tmp_path, "D")
    if unsafe_kind == "symlink":
        target = tmp_path / "target.json"
        target.write_text("operator evidence")
        path.symlink_to(target)
    else:
        path.mkdir()
    with pytest.raises(quality.QualityEvidenceError):
        quality.write_quality_sidecar_atomic(tmp_path, "D", item, ["turn_001.wav"])
    if unsafe_kind == "symlink":
        assert target.read_text() == "operator evidence"


def test_failed_atomic_replace_keeps_prior_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = evidence(tmp_path)
    path = quality.write_quality_sidecar_atomic(tmp_path, "D", item, ["turn_001.wav"])
    before = path.read_bytes()
    changed = deepcopy(item)
    changed["assessment"] = "reassessed"
    monkeypatch.setattr(
        quality.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("injected"))
    )
    with pytest.raises(OSError, match="injected"):
        quality.write_quality_sidecar_atomic(tmp_path, "D", changed, ["turn_001.wav"])
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".D_quality.*.tmp"))


def test_malformed_and_symlink_sidecars_do_not_load(tmp_path: Path) -> None:
    evidence(tmp_path)
    path = quality.quality_sidecar_path(tmp_path, "D")
    path.write_text('{"schema_version":"1.0","schema_version":"1.0"}')
    with pytest.raises(quality.QualityEvidenceError):
        quality.load_quality_sidecar(tmp_path, "D", ["turn_001.wav"])
    path.unlink()
    path.symlink_to(tmp_path / "missing.json")
    with pytest.raises(quality.QualityEvidenceError):
        quality.load_quality_sidecar(tmp_path, "D", ["turn_001.wav"])
