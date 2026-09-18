"""Frozen Controlled TTS v1 production-owned planner parity tests."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import subprocess
import sys
from collections.abc import Callable
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import pytest

from tts5703.controlled_tts import UpstreamValidationError, map_turn_to_higgs

EXPECTED_REFERENCE_COMMIT = "7b005135e148fad8a142f39dc8be3144c3e5adcc"
EXPECTED_CONTRACT_SHA256 = (
    "4e787a5363d01e5454a8031c59bb1d4bb454ee37d44ccf262d471c7ce30ba799"
)
PRODUCTION_CONTRACT = Path("src/tts5703/controlled_tts_v1.json")
REFERENCE_ROOT = Path("comp5703-tts-experiments")
REFERENCE_CONTRACT = (
    REFERENCE_ROOT / "mappings/controlled_tts_v1/controlled_tts_v1.json"
)
GOLDEN_PATH = Path("tests/fixtures/controlled_tts_v1_golden.json")
PAUSE_TOKEN = "<|prosody:pause|>"


def _turn(
    *,
    text: str = "A complete sentence. Another sentence.",
    rate: str = "normal",
    arousal: int = 2,
    affect: str = "neutral",
    pause_before: str = "none",
    pause_within: int = 0,
    hesitations: int = 0,
    best_effort: dict[str, Any] | None = None,
    speaker: str = "User",
) -> dict[str, Any]:
    acoustic: dict[str, Any] = {
        "required": {
            "rate": rate,
            "arousal": arousal,
            "affect": affect,
            "pause_before": pause_before,
            "pause_within": pause_within,
            "hesitations": hesitations,
        }
    }
    if best_effort is not None:
        acoustic["best_effort"] = best_effort
    return {
        "turn_id": 1,
        "speaker": speaker,
        "text": text,
        "acoustic": acoustic,
    }


def _plan_digest(plan: dict[str, Any]) -> str:
    encoded = json.dumps(
        plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _empty_best_effort() -> dict[str, Any]:
    return {
        "affect_fine": None,
        "volume": None,
        "flattened_affect": None,
        "events": [],
    }


def test_production_contract_has_frozen_identity_and_optional_reference_parity() -> (
    None
):
    production = PRODUCTION_CONTRACT.read_bytes()
    assert hashlib.sha256(production).hexdigest() == EXPECTED_CONTRACT_SHA256
    if REFERENCE_CONTRACT.is_file():
        assert production == REFERENCE_CONTRACT.read_bytes()


def test_committed_complete_plan_goldens() -> None:
    fixture = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert fixture["mapping_version"] == "controlled_tts_v1"
    assert fixture["provenance"] == {
        "source_repository": "ginkgoyin/comp5703-tts-experiments",
        "source_commit": EXPECTED_REFERENCE_COMMIT,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
    }
    for case in fixture["cases"]:
        source = deepcopy(case["turn"])
        plan = map_turn_to_higgs(
            case["turn"], dialogue_context=case.get("dialogue_context")
        )
        assert _plan_digest(plan) == case["plan_sha256"], case["id"]
        assert case["turn"] == source


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rate", "very_fast"),
        ("arousal", 4),
        ("arousal", True),
        ("affect", "joyful"),
        ("pause_before", "medium"),
        ("pause_within", -1),
        ("pause_within", True),
        ("hesitations", -1),
        ("hesitations", True),
        ("hesitations", 3),
    ],
)
def test_invalid_required_values_are_rejected_without_coercion(
    field: str, value: Any
) -> None:
    turn = _turn()
    turn["acoustic"]["required"][field] = value

    with pytest.raises(UpstreamValidationError):
        map_turn_to_higgs(turn)


@pytest.mark.parametrize(
    ("rate", "factor"), [("slow", 0.85), ("normal", None), ("fast", 1.15)]
)
def test_rate_is_postprocess_only(rate: str, factor: float | None) -> None:
    plan = map_turn_to_higgs(_turn(rate=rate))
    assert plan["higgs"]["native_rate_tokens"] == []
    assert plan["postprocess"]["atempo_factor"] == factor
    assert plan["postprocess"]["atempo"]["enabled"] is (factor is not None)


@pytest.mark.parametrize(
    ("arousal", "normalized", "token"),
    [
        (1, "low", "<|prosody:expressive_low|>"),
        (2, "medium", None),
        (3, "high", "<|prosody:expressive_high|>"),
    ],
)
def test_arousal_mapping(arousal: int, normalized: str, token: str | None) -> None:
    plan = map_turn_to_higgs(_turn(arousal=arousal))
    assert plan["normalized"]["arousal"] == normalized
    assert plan["realization"]["arousal"]["token"] == token


@pytest.mark.parametrize(
    ("affect", "token"),
    [
        ("neutral", None),
        ("sad", "<|emotion:helplessness|>"),
        ("anxious", "<|emotion:fear|>"),
        ("angry", "<|emotion:anger|>"),
        ("warm", "<|emotion:affection|>"),
    ],
)
def test_coarse_affect_mapping(affect: str, token: str | None) -> None:
    plan = map_turn_to_higgs(_turn(affect=affect))
    assert plan["realization"]["coarse_affect"]["token"] == token


def test_low_warm_uses_only_contentment_override() -> None:
    plan = map_turn_to_higgs(_turn(arousal=1, affect="warm"))
    assert plan["higgs"]["prefix_tokens"] == ["<|emotion:contentment|>"]
    assert "<|prosody:expressive_low|>" not in plan["higgs"]["model_input"]
    assert "<|emotion:affection|>" not in plan["higgs"]["model_input"]
    assert plan["combination_override"]["id"] == "low_warm_contentment"
    assert plan["validation_status"] is None


@pytest.mark.parametrize(
    ("pause_before", "milliseconds"),
    [("none", 0), ("short", 500), ("long", 900)],
)
def test_pause_before_plan(pause_before: str, milliseconds: int) -> None:
    plan = map_turn_to_higgs(_turn(pause_before=pause_before))
    assert plan["postprocess"]["pause_before_ms"] == milliseconds
    assert plan["postprocess"]["order"] == [
        "ffmpeg_atempo",
        "prepend_pause_before",
    ]


def test_hesitation_then_pause_planning_and_fallbacks_never_expose_p9() -> None:
    zero = map_turn_to_higgs(_turn(text="Help"))
    one = map_turn_to_higgs(_turn(text="Help", hesitations=1))
    two = map_turn_to_higgs(_turn(text="Help", hesitations=2))
    combined = map_turn_to_higgs(
        _turn(text="First. Second", hesitations=2, pause_within=1)
    )
    multiple = map_turn_to_higgs(_turn(text="Help", pause_within=3))

    assert zero["planner"]["hesitation"]["events"] == []
    assert one["higgs"]["text"] == "ummm, Help"
    assert two["higgs"]["text"] == "ummm... Help uhhh..."
    assert combined["planner"]["hesitation"]["text"] == (
        "First. ummm... Second uhhh..."
    )
    assert combined["higgs"]["text"].startswith(PAUSE_TOKEN)
    assert multiple["higgs"]["text"].count(PAUSE_TOKEN) == 3
    assert combined["higgs"]["synthesis_call_count"] == 1
    assert not re.search(
        r"(?:ummm,|ummm\.\.\.|uhhh\.\.\.)\s*" + re.escape(PAUSE_TOKEN),
        combined["higgs"]["text"],
    )
    assert any(
        warning["code"] == "hesitation_fallback" for warning in combined["warnings"]
    )
    assert any(
        warning["code"] == "pause_within_edge_fallback"
        for warning in combined["warnings"]
    )


def test_best_effort_statuses_and_warnings_are_preserved() -> None:
    requested = {
        "affect_fine": "hopeless",
        "volume": "soft",
        "flattened_affect": False,
        "events": ["door"],
    }
    plan = map_turn_to_higgs(_turn(affect="sad", best_effort=requested))
    fields = plan["best_effort"]["fields"]
    assert plan["requested"]["best_effort"] == requested
    assert fields["affect_fine"]["status"] == "jointly_realized"
    assert fields["volume"]["status"] == "unsupported"
    assert fields["flattened_affect"]["status"] == "not_realized"
    assert fields["events"]["status"] == "unsupported"
    assert "whisper" not in str(plan).lower()

    missing = map_turn_to_higgs(_turn())
    codes = [warning["code"] for warning in missing["warnings"]]
    assert "best_effort_fields_missing" in codes
    assert "speaker_id_unavailable" in codes
    assert "unvalidated_combination" in codes


def test_model_input_is_exact_prefix_plus_planned_text() -> None:
    plan = map_turn_to_higgs(
        _turn(
            text="I tried to explain. Then I stopped.",
            arousal=3,
            affect="angry",
            pause_within=1,
            hesitations=1,
        )
    )
    assert plan["higgs"]["model_input"] == (
        "".join(plan["higgs"]["prefix_tokens"]) + plan["higgs"]["text"]
    )
    assert plan["higgs"]["synthesis_call_count"] == 1


def _reference_mapper() -> Callable[..., dict[str, Any]]:
    if not REFERENCE_ROOT.is_dir():
        pytest.skip("optional controlled-TTS reference repository is absent")
    commit = subprocess.run(
        ["git", "-C", str(REFERENCE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_REFERENCE_COMMIT:
        pytest.skip(f"optional reference repository is at {commit}")
    sys.path.insert(0, str(REFERENCE_ROOT.resolve()))
    try:
        from mappings.controlled_tts_v1 import map_turn_to_higgs as reference
    finally:
        sys.path.pop(0)
    reference_file = Path(inspect.getfile(reference)).resolve()
    expected_package = (REFERENCE_ROOT / "mappings/controlled_tts_v1").resolve()
    assert reference_file.is_relative_to(expected_package), (
        f"reference adapter resolved outside expected package: {reference_file}"
    )
    return reference


def test_complete_plan_parity_with_expected_reference_checkout() -> None:
    reference = _reference_mapper()
    matrix = [
        _turn(
            rate=rate,
            arousal=arousal,
            affect=affect,
            pause_before=pause_before,
            best_effort=_empty_best_effort(),
        )
        for rate, arousal, affect, pause_before in product(
            ("slow", "normal", "fast"),
            (1, 2, 3),
            ("neutral", "sad", "anxious", "angry", "warm"),
            ("none", "short", "long"),
        )
    ]
    matrix.extend(
        [
            _turn(
                text="First sentence. A major clause, because it matters.",
                hesitations=hesitations,
                pause_within=within,
                best_effort=_empty_best_effort(),
            )
            for hesitations, within in product((0, 1, 2), (0, 1, 3))
        ]
    )
    matrix.extend(
        [
            _turn(
                text=text,
                hesitations=2,
                pause_within=2,
                best_effort={
                    "affect_fine": "hopeless",
                    "volume": "soft",
                    "flattened_affect": False,
                    "events": ["synthetic-event"],
                },
                affect="sad",
            )
            for text in (
                "Help",
                "I met John, Smith yesterday",
                "It would, not work today",
                "For e.g. this continues. Then it stops.",
            )
        ]
    )
    context = {
        "dialogue_id": "synthetic-dialogue",
        "scenario": {
            "speakers": {
                "caller": {"speaker_id": "C000"},
                "counsellor": {"speaker_id": "L000"},
            }
        },
        "known_issue": {"id": "synthetic-known-issue"},
    }
    for turn in matrix:
        assert map_turn_to_higgs(turn) == reference(turn)
    listener = _turn(speaker="Listener", best_effort=_empty_best_effort())
    assert map_turn_to_higgs(listener, dialogue_context=context) == reference(
        listener, dialogue_context=context
    )
