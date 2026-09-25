"""Offline, diagnostic-only measurements of rendered Higgs turn WAVs."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

import soundfile as sf

WINDOW_SEC = 0.25
RELATIVE_DROPS_DB = (6, 10, 14, 18)
KNOWN_BAD = Path("corpus_v1_000621/turn_012.wav")


def dbfs(power: float) -> float:
    return 10 * math.log10(power) if power > 0 else -math.inf


def display(value: float | None) -> str:
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return f"{value:.2f}"


def json_safe(value):
    """Keep strict JSON while preserving zero-energy and infinite drops."""
    if isinstance(value, float) and not math.isfinite(value):
        return "+inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def audio_metrics(path: Path) -> dict:
    windows = []
    with sf.SoundFile(path) as wav:
        sample_rate = wav.samplerate
        if sample_rate <= 0 or wav.frames == 0:
            raise ValueError("empty or invalid WAV")
        window_frames = max(1, round(sample_rate * WINDOW_SEC))
        for samples in wav.blocks(
            blocksize=window_frames, dtype="float64", always_2d=True
        ):
            # Average across frames and channels; a stereo file counts once in time.
            power = float((samples * samples).mean())
            windows.append((len(samples) / sample_rate, power, dbfs(power)))
        duration = wav.frames / sample_rate

    def tail(seconds: int) -> tuple[float, float]:
        selected = windows[-max(1, math.ceil(min(duration, seconds) / WINDOW_SEC)) :]
        power = sum(length * value for length, value, _ in selected) / sum(
            length for length, _, _ in selected
        )
        return power, dbfs(power)

    whole_power = sum(length * power for length, power, _ in windows) / duration
    # The first quarter (at most 5 s) avoids a long abnormal tail contaminating
    # the reference. Median window dB resists short pauses and brief peaks.
    body_count = max(1, min(math.ceil(len(windows) / 4), math.ceil(5 / WINDOW_SEC)))
    body_db = statistics.median(window[2] for window in windows[:body_count])
    tail_db = {str(seconds): tail(seconds)[1] for seconds in (2, 5)}
    tail_db["10"] = tail(10)[1] if duration >= 10 else None
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
            selected = windows[
                -max(1, math.ceil(min(duration, seconds) / WINDOW_SEC)) :
            ]
            return sum(db <= threshold_dbfs for _, _, db in selected) / len(selected)

        trailing = 0.0
        for length, _, db in reversed(windows):
            if db > threshold:
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
    }


def analyze(root: Path) -> tuple[list[dict], list[str]]:
    rows, warnings = [], []
    matched = set()
    metadata_paths = sorted(root.rglob("*_metadata.json"))
    if not metadata_paths:
        warnings.append("no dialogue metadata found")
    for metadata_path in metadata_paths:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            dialogue_id = metadata["dialogue_id"]
            turns = metadata["turns"]
            if not isinstance(dialogue_id, str) or not isinstance(turns, list):
                raise TypeError("invalid dialogue_id or turns")
        except (OSError, ValueError, KeyError, TypeError) as error:
            warnings.append(
                f"{metadata_path.relative_to(root)}: invalid metadata: {error}"
            )
            continue
        for turn in turns:
            try:
                identity = turn["source_identity"]
                ordinal = identity["ordinal"]
                name = turn["execution"]["turn_audio"]
                if (
                    type(ordinal) is not int
                    or ordinal < 1
                    or name != f"turn_{ordinal:03d}.wav"
                    or not re.fullmatch(r"turn_\d{3,}\.wav", name)
                ):
                    raise ValueError("invalid ordinal or turn WAV name")
                path = metadata_path.parent / name
                matched.add(path)
                if not path.is_file():
                    raise FileNotFoundError(name)
                higgs = turn["planned"]["higgs"]
                source_text, model_input = higgs["text"], higgs["model_input"]
                if not isinstance(source_text, str) or not isinstance(model_input, str):
                    raise TypeError("missing text or model input")
                measured = audio_metrics(path)
                chars, words = len(source_text), len(source_text.split())
                rows.append(
                    {
                        "dialogue_id": dialogue_id,
                        "ordinal": ordinal,
                        "source_turn_id": identity["source_turn_id"],
                        "wav_path": str(path.relative_to(root)),
                        "source_chars": chars,
                        "source_words": words,
                        "model_chars": len(model_input),
                        "model_words": len(model_input.split()),
                        "sec_per_char": measured["duration_sec"] / chars
                        if chars
                        else None,
                        "sec_per_word": measured["duration_sec"] / words
                        if words
                        else None,
                        **measured,
                    }
                )
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                warnings.append(
                    f"{metadata_path.relative_to(root)} turn {turn.get('source_identity', {}).get('ordinal', '?') if isinstance(turn, dict) else '?'}: {error}"
                )
    for path in sorted(root.rglob("turn_*.wav")):
        if path not in matched:
            warnings.append(
                f"{path.relative_to(root)}: WAV has no matching metadata turn"
            )
    rows.sort(key=lambda row: (row["wav_path"], row["ordinal"]))
    return rows, sorted(warnings)


def metric(row: dict, name: str) -> float | None:
    if name.startswith("trailing_"):
        return row["relative_thresholds"][name.split("_")[1]]["trailing_low_sec"]
    return row[name]


def summarize(rows: list[dict], name: str) -> dict:
    values = sorted(value for row in rows if (value := metric(row, name)) is not None)
    if not values:
        return {
            "count": 0,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(values),
        "median": statistics.median(values),
        **{f"p{p}": values[math.ceil(p * len(values) / 100) - 1] for p in (90, 95, 99)},
        "max": values[-1],
    }


def report(rows: list[dict], warnings: list[str]) -> None:
    names = ("duration_sec", "sec_per_word", "sec_per_char", "tail_5_drop_db") + tuple(
        f"trailing_{drop}_db" for drop in RELATIVE_DROPS_DB
    )
    known = next((row for row in rows if Path(row["wav_path"]) == KNOWN_BAD), None)
    print(f"Higgs audio diagnostics: {len(rows)} turns, {len(warnings)} warnings")
    print("250 ms decoded windows; body = median dBFS in first quarter (up to 5 s).")
    print("Tail drop = body dBFS minus final 5 s dBFS; +inf means digital silence.")
    print("Relative thresholds are diagnostic comparisons, not QC failure rules.")
    print("\nDistribution (all analyzed turns):")
    for name in names:
        stats = summarize(rows, name)
        print(
            f"  {name:19} n={stats['count']:4} median={display(stats['median']):>7} "
            f"p90={display(stats['p90']):>7} p95={display(stats['p95']):>7} "
            f"p99={display(stats['p99']):>7} max={display(stats['max']):>7}"
        )
    if known:
        print(
            f"\nKnown-bad comparison: {KNOWN_BAD} (observed 1024 cap-like duration: 40.680 s)"
        )
        for name in names:
            stats = summarize(rows, name)
            print(
                f"  {name:19} known={display(metric(known, name)):>7} "
                + " ".join(
                    f"{key}={display(stats[key]):>7}"
                    for key in ("median", "p90", "p95", "p99", "max")
                )
            )
        print(
            "  body/tail dBFS:",
            display(known["body_dbfs"]),
            *(
                f"{seconds}s={display(known['tail_dbfs'][seconds])}"
                for seconds in ("2", "5", "10")
            ),
        )
        for drop in RELATIVE_DROPS_DB:
            item = known["relative_thresholds"][str(drop)]
            print(
                f"  body-{drop:2} dB: final5 low={display(item['low_fraction_last_5_sec'])}, "
                f"final10 low={display(item['low_fraction_last_10_sec'])}, "
                f"trailing={display(item['trailing_low_sec'])} s"
            )
    print("\nTop 5 by metric (descending):")
    for name in names:
        ranked = sorted(
            (row for row in rows if metric(row, name) is not None),
            key=lambda row: (-metric(row, name), row["wav_path"]),
        )[:5]
        print(
            f"  {name}: "
            + ", ".join(
                f"{row['wav_path']}={display(metric(row, name))}" for row in ranked
            )
        )
    print(
        "\nPer turn: WAV | duration s | s/word | s/char | body dBFS | tail 5 dBFS | drop dB"
    )
    for row in rows:
        values = ("duration_sec", "sec_per_word", "sec_per_char", "body_dbfs")
        print(
            "  "
            + " | ".join(
                (
                    row["wav_path"],
                    *(display(row[key]) for key in values),
                    display(row["tail_dbfs"]["5"]),
                    display(row["tail_5_drop_db"]),
                )
            )
        )
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if not args.output_root.is_dir():
        parser.error(f"output root does not exist: {args.output_root}")
    rows, warnings = analyze(args.output_root)
    report(rows, warnings)
    if args.json_out:
        with args.json_out.open("x", encoding="utf-8") as output:
            json.dump(
                json_safe({"turns": rows, "warnings": warnings}), output, indent=2
            )
            output.write("\n")


if __name__ == "__main__":
    main()
