"""Higgs termination-sanity measurements and durable per-dialogue evidence."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import stat
import statistics
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import soundfile as sf

WINDOW_SEC = 0.25
RELATIVE_DROPS_DB = (6, 10, 14, 18)
POLICY_ID = "higgs_audio_quality_v1"
QUALITY_SCHEMA_VERSION = "1.0"
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_TURN_WAV = re.compile(r"turn_(?:00[1-9]|0[1-9][0-9]|[1-9][0-9]{2,})\.wav")

# These definitions are part of the policy, not documentation. Change the ID
# whenever a signal calculation or decision rule changes.
_POLICY_PARAMETERS = {
    "decoder": {
        "library": "soundfile",
        "dtype": "float64",
        "channels": "all_equal_weight",
    },
    "window": {
        "seconds": WINDOW_SEC,
        "frames": "max(1, round(sample_rate * seconds))",
        "rounding": "python_half_even",
        "final_block": "unpad_actual_frame_count",
        "power": "mean_of_squared_samples_across_frames_and_channels",
        "dbfs": "10*log10(power); zero=-inf",
    },
    "body": {
        "window_count": "max(1,min(ceil(total_windows/4),ceil(5.0/window_seconds)))",
        "aggregate": "median_linear_window_power_then_dbfs",
    },
    "tail": {
        "seconds": 5.0,
        "selection": "reverse_accumulated_actual_window_duration",
        "boundary": "full_block_power_weighted_by_clipped_overlap_duration",
        "aggregate": "duration_weighted_linear_power_then_dbfs",
        "remaining_epsilon_sec": 1e-12,
    },
    "relative": {
        "body_reference_minus_db": 10.0,
        "window_comparison": "less_than_or_equal",
        "trailing": "sum_actual_window_durations_until_first_higher_window",
    },
    "predicate": {
        "minimum_tail_drop_db": 10.0,
        "minimum_trailing_low_energy_sec": 3.0,
        "comparisons": "greater_than_or_equal",
        "operator": "and",
    },
    "exact_digital_zero": "reject_if_every_decoded_sample_equals_zero",
    "required_metrics": {
        "unavailable_or_nan": "indeterminate",
        "negative_infinite_tail_drop": "indeterminate",
        "positive_infinite_tail_drop": "valid_satisfies_minimum_drop",
        "nonfinite_trailing_duration": "indeterminate",
    },
    "diagnostic_tail_seconds": [2, 5, 10],
    "diagnostic_relative_drops_db": list(RELATIVE_DROPS_DB),
}


class QualityEvidenceError(ValueError):
    """Present quality evidence cannot be trusted or safely replaced."""


def _policy_hash(policy_id: str, parameters: dict[str, Any]) -> str:
    payload = json.dumps(
        {"id": policy_id, "parameters": parameters},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def policy_identity() -> dict[str, Any]:
    """Return the complete frozen policy and its canonical SHA256."""
    parameters = deepcopy(_POLICY_PARAMETERS)
    return {
        "id": POLICY_ID,
        "sha256": _policy_hash(POLICY_ID, parameters),
        "parameters": parameters,
    }


def dbfs(power: float) -> float:
    return 10 * math.log10(power) if power > 0 else -math.inf


def json_safe(value: Any) -> Any:
    """Encode meaningful infinite dBFS values as strict JSON strings."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError("NaN is not a valid quality metric")
        return "+inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def audio_metrics(path: Path) -> dict[str, Any]:
    """Measure decoded WAVs in unpadded, fixed-frame windows."""
    windows = []
    exact_zero = True
    with sf.SoundFile(path) as wav:
        sample_rate = wav.samplerate
        if sample_rate <= 0 or wav.frames == 0:
            raise ValueError("empty or invalid WAV")
        window_frames = max(1, round(sample_rate * WINDOW_SEC))
        for samples in wav.blocks(
            blocksize=window_frames, dtype="float64", always_2d=True
        ):
            exact_zero &= not bool(samples.any())
            # Average across frames and channels; stereo counts once in time.
            power = float((samples * samples).mean())
            if not math.isfinite(power):
                raise ValueError("nonfinite decoded audio power")
            windows.append((len(samples) / sample_rate, power, dbfs(power)))
        duration = wav.frames / sample_rate

    def tail_windows(seconds: int) -> list[tuple[float, float, float]]:
        remaining = min(duration, seconds)
        selected = []
        for length, power, level in reversed(windows):
            if remaining <= 1e-12:
                break
            overlap = min(length, remaining)
            selected.append((overlap, power, level))
            remaining -= overlap
        return selected

    def tail(seconds: int) -> float:
        selected = tail_windows(seconds)
        power = sum(length * value for length, value, _ in selected) / sum(
            length for length, _, _ in selected
        )
        return dbfs(power)

    whole_power = sum(length * power for length, power, _ in windows) / duration
    body_count = max(1, min(math.ceil(len(windows) / 4), math.ceil(5 / WINDOW_SEC)))
    body_db = dbfs(statistics.median(window[1] for window in windows[:body_count]))
    tail_db = {str(seconds): tail(seconds) for seconds in (2, 5)}
    tail_db["10"] = tail(10) if duration >= 10 else None
    tail_drop = body_db - tail_db["5"] if math.isfinite(body_db) else None
    relative = {}
    for drop in RELATIVE_DROPS_DB:
        if not math.isfinite(body_db):
            relative[str(drop)] = {
                "threshold_dbfs": None,
                "low_fraction_last_5_sec": None,
                "low_fraction_last_10_sec": None,
                "trailing_low_sec": None,
            }
            continue
        threshold = body_db - drop

        def fraction(seconds: int, threshold_dbfs: float = threshold) -> float:
            selected = tail_windows(seconds)
            return sum(
                length for length, _, level in selected if level <= threshold_dbfs
            ) / sum(length for length, _, _ in selected)

        trailing = 0.0
        for length, _, level in reversed(windows):
            if level > threshold:
                break
            trailing += length
        relative[str(drop)] = {
            "threshold_dbfs": threshold,
            "low_fraction_last_5_sec": fraction(5),
            "low_fraction_last_10_sec": fraction(10) if duration >= 10 else None,
            "trailing_low_sec": trailing,
        }
    return {
        "duration_sec": duration,
        "whole_dbfs": dbfs(whole_power),
        "body_dbfs": body_db,
        "tail_dbfs": tail_db,
        "tail_5_drop_db": tail_drop,
        "relative_thresholds": relative,
        "exact_digital_zero": exact_zero,
    }


def assess_turn(path: Path) -> dict[str, Any]:
    """Return a quality outcome; metric/decoder errors are indeterminate."""
    try:
        metrics = audio_metrics(path)
        if metrics["exact_digital_zero"]:
            return {
                "outcome": "reject",
                "issues": [{"code": "exact_digital_silence"}],
                "metrics": metrics,
            }
        drop = metrics["tail_5_drop_db"]
        trailing = metrics["relative_thresholds"]["10"]["trailing_low_sec"]
        if (
            drop is None
            or trailing is None
            or (not math.isfinite(drop) and drop != math.inf)
            or not math.isfinite(trailing)
        ):
            return {
                "outcome": "indeterminate",
                "issues": [{"code": "metrics_unavailable"}],
                "metrics": None,
            }
        if drop >= 10.0 and trailing >= 3.0:
            return {
                "outcome": "reject",
                "issues": [{"code": "abnormal_tail"}],
                "metrics": metrics,
            }
        return {"outcome": "pass", "issues": [], "metrics": metrics}
    except Exception as error:  # noqa: BLE001 - quality uncertainty must not pass
        return {
            "outcome": "indeterminate",
            "issues": [{"code": "metrics_unavailable", "message": str(error)}],
            "metrics": None,
        }


def wav_sha256(path: Path) -> str:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise OSError(f"Turn WAV is not a regular file: {path}")
        return hashlib.file_digest(handle, "sha256").hexdigest()


def quality_sidecar_path(out_dir: Path, dialogue_id: str) -> Path:
    return out_dir / f"{dialogue_id}_quality.json"


def assess_dialogue_turns(
    out_dir: Path,
    dialogue_id: str,
    render_fingerprint: str,
    metadata_turns: list[dict[str, Any]],
    *,
    assessment: str,
) -> dict[str, Any]:
    """Assess every metadata-listed turn and hash its raw WAV bytes."""
    issues: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    outcomes = set()
    for turn in metadata_turns:
        name = turn["execution"]["turn_audio"]
        ordinal = turn["source_identity"]["ordinal"]
        if name != f"turn_{ordinal:03d}.wav" or not _TURN_WAV.fullmatch(name):
            raise QualityEvidenceError("invalid metadata turn WAV name")
        path = out_dir / name
        hashes[name] = wav_sha256(path)
        result = assess_turn(path)
        outcomes.add(result["outcome"])
        for issue in result["issues"]:
            issues.append(
                {
                    "ordinal": ordinal,
                    "source_turn_id": turn["source_identity"]["source_turn_id"],
                    **issue,
                    **(
                        {"metrics": json_safe(result["metrics"])}
                        if result["metrics"] is not None
                        else {}
                    ),
                }
            )
    outcome = (
        "indeterminate"
        if "indeterminate" in outcomes
        else "reject"
        if "reject" in outcomes
        else "pass"
    )
    evidence = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "dialogue_id": dialogue_id,
        "render_fingerprint": render_fingerprint,
        "policy": policy_identity(),
        "outcome": outcome,
        "assessment": assessment,
        "issues": issues,
        "turn_wav_sha256": hashes,
    }
    validate_quality_sidecar(
        evidence,
        dialogue_id,
        [turn["execution"]["turn_audio"] for turn in metadata_turns],
    )
    return evidence


def validate_quality_sidecar(
    evidence: Any, dialogue_id: str, expected_turn_names: list[str]
) -> dict[str, Any]:
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema_version") != QUALITY_SCHEMA_VERSION
    ):
        raise QualityEvidenceError("invalid quality sidecar schema")
    if evidence.get("dialogue_id") != dialogue_id:
        raise QualityEvidenceError("quality sidecar dialogue mismatch")
    fingerprint = evidence.get("render_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise QualityEvidenceError("missing quality render fingerprint")
    policy = evidence.get("policy")
    if (
        not isinstance(policy, dict)
        or not isinstance(policy.get("id"), str)
        or not policy["id"].strip()
        or not isinstance(policy.get("parameters"), dict)
    ):
        raise QualityEvidenceError("invalid quality policy")
    if policy.get("sha256") != _policy_hash(policy["id"], policy["parameters"]):
        raise QualityEvidenceError("quality policy hash mismatch")
    if evidence.get("outcome") not in {"pass", "reject", "indeterminate"}:
        raise QualityEvidenceError("invalid quality outcome")
    if evidence.get("assessment") not in {"new_render", "reassessed"}:
        raise QualityEvidenceError("invalid quality assessment")
    issues = evidence.get("issues")
    if (
        not isinstance(issues, list)
        or any(
            not isinstance(issue, dict) or not isinstance(issue.get("code"), str)
            for issue in issues
        )
        or (evidence["outcome"] == "pass") != (not issues)
    ):
        raise QualityEvidenceError("invalid quality issues")
    hashes = evidence.get("turn_wav_sha256")
    if (
        not isinstance(hashes, dict)
        or not expected_turn_names
        or len(expected_turn_names) != len(set(expected_turn_names))
        or any(not _TURN_WAV.fullmatch(name) for name in expected_turn_names)
        or set(hashes) != set(expected_turn_names)
        or any(
            not isinstance(digest, str) or not _HEX_SHA256.fullmatch(digest)
            for digest in hashes.values()
        )
    ):
        raise QualityEvidenceError("incomplete quality turn WAV hashes")
    return evidence


def load_quality_sidecar(
    out_dir: Path, dialogue_id: str, expected_turn_names: list[str]
) -> dict[str, Any] | None:
    path = quality_sidecar_path(out_dir, dialogue_id)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    except OSError as error:
        raise QualityEvidenceError(f"cannot read quality sidecar: {error}") from error
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise QualityEvidenceError("quality sidecar is not a regular file")
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            evidence = json.load(
                handle,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
    except (OSError, ValueError, TypeError) as error:
        raise QualityEvidenceError(f"invalid quality sidecar: {error}") from error
    finally:
        if fd >= 0:
            os.close(fd)
    return validate_quality_sidecar(evidence, dialogue_id, expected_turn_names)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise QualityEvidenceError(f"duplicate quality key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise QualityEvidenceError(f"invalid quality JSON constant {value}")


def write_quality_sidecar_atomic(
    out_dir: Path,
    dialogue_id: str,
    evidence: dict[str, Any],
    expected_turn_names: list[str],
) -> Path:
    validate_quality_sidecar(evidence, dialogue_id, expected_turn_names)
    path = quality_sidecar_path(out_dir, dialogue_id)
    try:
        existing = path.lstat()
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISREG(existing.st_mode):
            raise QualityEvidenceError("quality sidecar path is not a regular file")
    payload = (
        json.dumps(
            evidence, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=out_dir,
        prefix=f".{dialogue_id}_quality.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temp_path = Path(temporary.name)
        try:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp_path, path)
        directory_fd = os.open(out_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            try:
                os.fsync(directory_fd)
            except OSError as error:
                if error.errno not in {errno.EINVAL, errno.ENOTSUP}:
                    raise
        finally:
            os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)
    return path


def verify_turn_hashes(out_dir: Path, evidence: dict[str, Any]) -> str | None:
    """Return an integrity issue, or None when every stored WAV hash matches."""
    for name, expected in evidence["turn_wav_sha256"].items():
        try:
            if wav_sha256(out_dir / name) != expected:
                return f"Turn WAV hash changed: {name}"
        except OSError as error:
            return f"Cannot hash turn WAV {name}: {error}"
    return None
