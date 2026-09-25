"""Offline, diagnostic-only measurements of rendered Higgs turn WAVs."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections.abc import Mapping
from pathlib import Path

from tts5703.audio_quality import RELATIVE_DROPS_DB, audio_metrics, json_safe

KNOWN_BAD = Path("corpus_v1_000621/turn_012.wav")


def display(value: float | None) -> str:
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return f"{value:.2f}"


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
            identity = turn.get("source_identity") if isinstance(turn, dict) else None
            warning_ordinal = (
                identity.get("ordinal", "?") if isinstance(identity, Mapping) else "?"
            )
            try:
                if not isinstance(identity, Mapping):
                    raise TypeError("invalid source_identity")
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
                planned_text, model_input = higgs["text"], higgs["model_input"]
                if not isinstance(planned_text, str) or not isinstance(
                    model_input, str
                ):
                    raise TypeError("missing text or model input")
                measured = audio_metrics(path)
                chars, words = len(planned_text), len(planned_text.split())
                rows.append(
                    {
                        "dialogue_id": dialogue_id,
                        "ordinal": ordinal,
                        "source_turn_id": identity["source_turn_id"],
                        "wav_path": str(path.relative_to(root)),
                        "planned_chars": chars,
                        "planned_words": words,
                        "model_chars": len(model_input),
                        "model_words": len(model_input.split()),
                        "sec_per_planned_char": measured["duration_sec"] / chars
                        if chars
                        else None,
                        "sec_per_planned_word": measured["duration_sec"] / words
                        if words
                        else None,
                        **measured,
                    }
                )
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                warnings.append(
                    f"{metadata_path.relative_to(root)} turn {warning_ordinal}: {error}"
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
    names = (
        "duration_sec",
        "sec_per_planned_word",
        "sec_per_planned_char",
        "tail_5_drop_db",
    ) + tuple(f"trailing_{drop}_db" for drop in RELATIVE_DROPS_DB)
    known = next((row for row in rows if Path(row["wav_path"]) == KNOWN_BAD), None)
    print(f"Higgs audio diagnostics: {len(rows)} turns, {len(warnings)} warnings")
    print(
        "250 ms decoded windows; body = dBFS of median power in first quarter (up to 5 s)."
    )
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
            f"\nKnown-bad comparison: {KNOWN_BAD} "
            f"(observed cap-like duration: {display(known['duration_sec'])} s)"
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
        "\nPer turn: WAV | duration s | s/planned word | s/planned char | body dBFS | tail 5 dBFS | drop dB"
    )
    for row in rows:
        values = (
            "duration_sec",
            "sec_per_planned_word",
            "sec_per_planned_char",
            "body_dbfs",
        )
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
