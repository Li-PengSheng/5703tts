from __future__ import annotations

import pytest

from tts5703.backend_errors import BackendControlError
from tts5703.cosyvoice_controls import (
    COSYVOICE_CONTROL_MAPPING_NAME,
    COSYVOICE_CONTROL_MAPPING_STATUS,
    COSYVOICE_CONTROL_MAPPING_VERSION,
    build_cosyvoice_instruction,
    rate_to_cosyvoice_speed,
    resolve_cosyvoice_controls,
)


@pytest.mark.parametrize(
    ("rate", "speed"), [("slow", 0.8), ("normal", 1.0), ("fast", 1.2)]
)
def test_final_rate_mapping(rate: str, speed: float) -> None:
    assert rate_to_cosyvoice_speed(rate) == speed


@pytest.mark.parametrize("rate", ["+20%", "-20%", "very_fast"])
def test_non_final_rates_fail_closed(rate: str) -> None:
    with pytest.raises(BackendControlError, match="rate mapping"):
        rate_to_cosyvoice_speed(rate)


@pytest.mark.parametrize("arousal", ["low", "medium", "high"])
@pytest.mark.parametrize("affect", ["neutral", "sad", "anxious", "angry", "warm"])
def test_final_instruction_domains(arousal: str, affect: str) -> None:
    instruction = build_cosyvoice_instruction(arousal, affect)
    assert instruction is not None
    assert instruction.endswith("<|endofprompt|>")
    assert instruction.count("<|endofprompt|>") == 1


@pytest.mark.parametrize(
    ("arousal", "affect", "field"),
    [("frantic", "neutral", "arousal"), ("medium", "distressed", "coarse_affect")],
)
def test_controls_outside_final_contract_fail_closed(
    arousal: str, affect: str, field: str
) -> None:
    with pytest.raises(BackendControlError, match=field):
        build_cosyvoice_instruction(arousal, affect)


def test_cached_resolution_identity_and_values() -> None:
    resolution = resolve_cosyvoice_controls("slow", "high", "anxious")

    assert resolution["mapping"] == {
        "name": COSYVOICE_CONTROL_MAPPING_NAME,
        "version": COSYVOICE_CONTROL_MAPPING_VERSION,
        "status": COSYVOICE_CONTROL_MAPPING_STATUS,
    }
    assert resolution["speaking_rate"] == {
        "requested": "slow",
        "method": "numeric_speed",
        "speed": 0.8,
    }
    assert resolution["inference_mode"] == "instruct2"
    assert "anxious" in resolution["resolved_instruction"]
    assert "high vocal activation" in resolution["resolved_instruction"]
