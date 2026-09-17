"""Transparent deterministic text planners for provisional v1 controls."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_PUNCTUATION = ".!?;:—,"


def _word_before(text: str, index: int) -> str:
    match = re.search(r"([A-Za-z]+)\W*$", text[:index])
    return match.group(1).lower() if match else ""


def _word_after(text: str, index: int) -> str:
    match = re.match(r"\s*([A-Za-z]+)", text[index:])
    return match.group(1).lower() if match else ""


def _raw_word_before(text: str, index: int) -> str:
    match = re.search(r"([A-Za-z]+)\W*$", text[:index])
    return match.group(1) if match else ""


def _raw_word_after(text: str, index: int) -> str:
    match = re.match(r"\s*([A-Za-z]+)", text[index:])
    return match.group(1) if match else ""


def _filler_words(contract: dict[str, Any]) -> set[str]:
    return {
        word.lower()
        for value in contract["controls"]["hesitation_count"]["values"].values()
        for event in value["events"]
        for word in re.findall(r"[A-Za-z]+", event)
    }


def _unsafe_boundary(
    text: str,
    offset: int,
    after_index: int,
    contract: dict[str, Any],
    *,
    mark: str | None = None,
) -> bool:
    """Reject the small, explicit set of frozen protected boundary forms."""
    config = contract["planner_contracts"]
    before, after = _word_before(text, offset), _word_after(text, after_index)
    raw_before = _raw_word_before(text, offset)
    raw_after = _raw_word_after(text, after_index)
    if not before or not after:
        return True
    if before in _filler_words(contract) or after in _filler_words(contract):
        return True
    if (
        mark in {",", "whitespace"}
        and raw_before[:1].isupper()
        and raw_after[:1].isupper()
    ):
        return True
    if mark in {",", "whitespace"} and (len(raw_before) == 1 or len(raw_after) == 1):
        return True
    if before in config["auxiliaries"] and after in config["negations"]:
        return True
    if before in config["negations"] and after in config["auxiliaries"]:
        return True
    if any(
        before == phrase[0] and after == phrase[-1]
        for phrase in config["fixed_expressions"]
    ):
        return True
    window = text[max(0, offset - 32) : min(len(text), after_index + 32)]
    return "<|" in window or "|>" in window


def _ends_with_configured_abbreviation(
    text: str, offset: int, abbreviations: set[str]
) -> bool:
    """Return whether text through this full stop ends in an abbreviation."""
    prefix = text[: offset + 1].lower()
    for abbreviation in abbreviations:
        candidate = abbreviation + "."
        if not prefix.endswith(candidate):
            continue
        preceding = len(prefix) - len(candidate) - 1
        if preceding < 0 or not prefix[preceding].isalnum():
            return True
    return False


def _natural_boundaries(text: str, contract: dict[str, Any]) -> list[dict[str, Any]]:
    config = contract["planner_contracts"]
    abbreviations = set(config["sentence_abbreviations"])
    introducers = set(config["clause_introducers"])
    result: list[dict[str, Any]] = []
    for match in re.finditer(r"([.!?;:—,])(\s+)(?=\S)", text):
        mark, offset = match.group(1), match.start(1)
        before = _word_before(text, offset)
        if mark == "." and (
            before in abbreviations
            or _ends_with_configured_abbreviation(text, offset, abbreviations)
            or bool(re.search(r"\d$", text[:offset]))
            or (len(before) == 1 and text[match.end() :].lstrip()[:1].isupper())
        ):
            continue
        if _unsafe_boundary(text, offset, match.end(1), contract, mark=mark):
            continue
        after = _word_after(text, match.end(1))
        if (mark == "," and after in introducers) or mark in ";:—":
            tier = 2
        elif mark in ".!?":
            tier = 1
        else:
            tier = 3
        result.append(
            {
                "tier": tier,
                "offset": offset + 1,
                "kind": {1: "sentence", 2: "major_clause", 3: "comma"}[tier],
            }
        )
    return sorted(result, key=lambda item: (item["tier"], item["offset"]))


def _whitespace_boundaries(text: str, contract: dict[str, Any]) -> list[dict[str, Any]]:
    """Return conservative, left-to-right fallback boundaries for native pauses."""
    result: list[dict[str, Any]] = []
    for match in re.finditer(r"\s+(?=\S)", text):
        offset = match.start()
        if offset and text[offset - 1] in _PUNCTUATION:
            continue
        if _unsafe_boundary(text, offset, match.end(), contract, mark="whitespace"):
            continue
        result.append({"offset": offset, "kind": "conservative_whitespace"})
    return result


def _insert(text: str, edits: Iterable[tuple[int, str]]) -> str:
    result = text
    for offset, value in sorted(edits, reverse=True):
        result = result[:offset] + value + result[offset:]
    return result


def _event_spans(
    edits: list[tuple[int, str]], events: list[str]
) -> list[tuple[int, int]]:
    """Calculate inserted event spans from source offsets, never text searching."""
    spans: list[tuple[int, int]] = []
    prior_length = 0
    for (offset, edit), event in sorted(
        zip(edits, events), key=lambda item: item[0][0]
    ):
        start = offset + prior_length + len(edit) - len(event)
        spans.append((start, start + len(event)))
        prior_length += len(edit)
    return spans


def plan_hesitations(text: str, count: int, contract: dict[str, Any]) -> dict[str, Any]:
    values = contract["controls"]["hesitation_count"]["values"]
    events = list(values[str(count)]["events"])
    if not events:
        return {
            "text": text,
            "requested_count": 0,
            "inserted_count": 0,
            "events": [],
            "used_fallback": False,
            "status": values["0"]["status"],
            "warnings": [],
            "reserved_spans": [],
        }
    selected = _natural_boundaries(text, contract)[:count]
    if count == 2:
        selected = sorted(selected, key=lambda item: item["offset"])
    fallback = len(selected) < count
    warnings: list[dict[str, Any]] = []
    if count == 1 and not selected:
        output = events[0] + " " + text
        spans = [(0, len(events[0]))]
        placements = [{"event": events[0], "placement": "prefix_fallback"}]
    elif count == 2 and len(selected) == 1:
        edits = [(selected[0]["offset"], " " + events[0])]
        output = _insert(text, edits) + " " + events[1]
        spans = _event_spans(edits, events[:1]) + [
            (len(output) - len(events[1]), len(output))
        ]
        placements = [
            {"event": events[0], "placement": selected[0]["kind"]},
            {"event": events[1], "placement": "terminal_fallback"},
        ]
    elif count == 2 and not selected:
        output = events[0] + " " + text + " " + events[1]
        spans = [
            (0, len(events[0])),
            (len(output) - len(events[1]), len(output)),
        ]
        placements = [
            {"event": events[0], "placement": "prefix_fallback"},
            {"event": events[1], "placement": "terminal_fallback"},
        ]
    else:
        edits = [
            (candidate["offset"], " " + event)
            for candidate, event in zip(selected, events)
        ]
        output = _insert(text, edits)
        spans = _event_spans(edits, events)
        placements = [
            {"event": event, "placement": candidate["kind"]}
            for candidate, event in zip(selected, events)
        ]
    if fallback:
        warnings.append(
            {
                "code": "hesitation_fallback",
                "control": "hesitation_count",
                "requested_count": count,
                "message": (
                    "Insufficient safe natural boundaries; documented "
                    "deterministic fallback used."
                ),
            }
        )
    return {
        "text": output,
        "requested_count": count,
        "inserted_count": len(events),
        "events": placements,
        "used_fallback": fallback,
        "status": values[str(count)]["status"],
        "warnings": warnings,
        "reserved_spans": spans,
    }


def _adjacent_to_reserved(offset: int, spans: list[tuple[int, int]]) -> bool:
    return any(start - 1 <= offset <= end + 1 for start, end in spans)


def _assert_no_filler_pause_pattern(text: str, contract: dict[str, Any]) -> None:
    token = re.escape(contract["controls"]["pause_within_count"]["token"])
    for event in _filler_words(contract):
        if re.search(rf"\b{re.escape(event)}(?:,|\.\.\.)?\s*{token}", text):
            raise AssertionError(
                "P9 diagnostic-only filler/pause realization must not be exposed"
            )


def plan_within_pauses(
    text: str,
    count: int,
    reserved_spans: list[tuple[int, int]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    control = contract["controls"]["pause_within_count"]
    token = control["token"]
    if count == 0:
        return {
            "text": text,
            "requested_count": 0,
            "inserted_count": 0,
            "events": [],
            "used_fallback": False,
            "status": control["zero_status"],
            "warnings": [],
        }
    natural = [
        item
        for item in _natural_boundaries(text, contract)
        if not _adjacent_to_reserved(item["offset"], reserved_spans)
    ]
    selected = natural[:count]
    remaining = count - len(selected)
    if remaining:
        whitespace = [
            item
            for item in _whitespace_boundaries(text, contract)
            if not _adjacent_to_reserved(item["offset"], reserved_spans)
        ]
        selected.extend(whitespace[:remaining])
    output = _insert(text, [(item["offset"], token) for item in selected])
    remaining = count - len(selected)
    warnings: list[dict[str, Any]] = []
    events = [{"placement": item["kind"]} for item in selected]
    if remaining:
        terminal_blocked = any(end >= len(text) for _start, end in reserved_spans)
        placement = (
            "leading_edge_fallback" if terminal_blocked else "terminal_edge_fallback"
        )
        output = (
            token * remaining + output
            if terminal_blocked
            else output + token * remaining
        )
        events.extend({"placement": placement} for _ in range(remaining))
        warnings.append(
            {
                "code": "pause_within_edge_fallback",
                "control": "pause_within_count",
                "requested_count": count,
                "fallback_count": remaining,
                "placement": placement,
                "message": (
                    "Insufficient safe natural or whitespace boundaries; "
                    "remaining native pause tokens use documented edge fallback."
                ),
            }
        )
    _assert_no_filler_pause_pattern(output, contract)
    if output.count(token) != count:
        raise AssertionError(
            "pause_within planner did not preserve the requested native token count"
        )
    return {
        "text": output,
        "requested_count": count,
        "inserted_count": count,
        "events": events,
        "used_fallback": bool(remaining),
        "status": control["positive_status"],
        "warnings": warnings,
    }
