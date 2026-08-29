"""Offline tests for the central engine acoustic-control capability registry."""

import pytest

from tts5703 import engine_capabilities as registry

EXPECTED_COSYVOICE_SUPPORT = {
    "rate": "model_control",
    "pause_before_ms": "pipeline_timing",
    "pause_after_ms": "pipeline_timing",
    "arousal": "provisional_model_control",
    "coarse_affect": "provisional_model_control",
    "emotion": "unsupported",
    "paralinguistic_events": "unsupported",
}
EXPECTED_KOKORO_SUPPORT = {
    "rate": "model_control",
    "pause_before_ms": "pipeline_timing",
    "pause_after_ms": "pipeline_timing",
    "arousal": "unsupported",
    "coarse_affect": "unsupported",
    "emotion": "unsupported",
    "paralinguistic_events": "unsupported",
}


def test_cosyvoice_capabilities_match_declared_semantics() -> None:
    assert registry.control_support("cosyvoice") == EXPECTED_COSYVOICE_SUPPORT


def test_kokoro_capabilities_match_declared_semantics() -> None:
    assert registry.control_support("kokoro") == EXPECTED_KOKORO_SUPPORT


def test_declared_engines_are_only_the_current_backends() -> None:
    assert registry.declared_engines() == ["cosyvoice", "kokoro"]
    assert registry.has_declared_capabilities("kokoro")
    assert not registry.has_declared_capabilities("edge_tts")
    assert not registry.has_declared_capabilities(None)


def test_capability_vocabulary_is_the_accepted_set() -> None:
    assert registry.CAPABILITY_VOCABULARY == {
        "model_control",
        "provisional_model_control",
        "pipeline_timing",
        "unsupported",
    }


def test_every_declared_support_value_uses_the_accepted_vocabulary() -> None:
    for engine, capabilities in registry.ENGINE_CAPABILITIES.items():
        assert set(capabilities) == set(registry.ACOUSTIC_CONTROL_FIELDS), engine
        for field, description in capabilities.items():
            assert description["support"] in registry.CAPABILITY_VOCABULARY, (
                engine,
                field,
            )


def test_capabilities_are_returned_as_an_isolated_copy() -> None:
    capabilities = registry.engine_capabilities("kokoro")
    capabilities["arousal"]["support"] = "model_control"

    assert registry.engine_capabilities("kokoro")["arousal"]["support"] == "unsupported"


def test_undeclared_engine_is_rejected_with_declared_engines_listed() -> None:
    with pytest.raises(
        registry.UnknownEngineCapabilityError,
        match=r"engine 'edge_tts'.*\['cosyvoice', 'kokoro'\]",
    ):
        registry.engine_capabilities("edge_tts")


class _Turn:
    def __init__(self, **fields: object) -> None:
        defaults: dict[str, object] = {
            "rate": "normal",
            "pause_before_ms": 0,
            "pause_after_ms": 500,
            "arousal": None,
            "coarse_affect": None,
            "emotion": None,
            "paralinguistic_events": [],
        }
        for field, value in {**defaults, **fields}.items():
            setattr(self, field, value)


def test_requested_acoustic_spec_snapshots_every_control_field() -> None:
    spec = registry.requested_acoustic_spec(
        _Turn(rate="slow", arousal="high", coarse_affect="distressed")
    )

    assert spec == {
        "rate": "slow",
        "pause_before_ms": 0,
        "pause_after_ms": 500,
        "arousal": "high",
        "coarse_affect": "distressed",
        "emotion": None,
        "paralinguistic_events": [],
    }


def test_kokoro_ignores_requested_arousal_and_coarse_affect() -> None:
    spec = registry.requested_acoustic_spec(
        _Turn(arousal="high", coarse_affect="distressed")
    )

    assert registry.ignored_requested_controls("kokoro", spec) == [
        "arousal",
        "coarse_affect",
    ]


def test_null_controls_are_not_reported_as_ignored() -> None:
    spec = registry.requested_acoustic_spec(_Turn())

    assert registry.ignored_requested_controls("kokoro", spec) == []


def test_pipeline_timing_controls_are_never_reported_as_ignored() -> None:
    spec = registry.requested_acoustic_spec(
        _Turn(pause_before_ms=800, pause_after_ms=1000)
    )

    assert registry.ignored_requested_controls("kokoro", spec) == []
    assert registry.ignored_requested_controls("cosyvoice", spec) == []


def test_empty_paralinguistic_events_are_not_ignored_but_populated_ones_are() -> None:
    empty = registry.requested_acoustic_spec(_Turn(paralinguistic_events=[]))
    populated = registry.requested_acoustic_spec(_Turn(paralinguistic_events=["sigh"]))

    assert registry.ignored_requested_controls("cosyvoice", empty) == []
    assert registry.ignored_requested_controls("cosyvoice", populated) == [
        "paralinguistic_events"
    ]


def test_cosyvoice_provisional_controls_are_not_reported_as_ignored() -> None:
    spec = registry.requested_acoustic_spec(
        _Turn(arousal="high", coarse_affect="distressed", emotion="fearful")
    )

    assert registry.ignored_requested_controls("cosyvoice", spec) == ["emotion"]
    support = registry.control_support("cosyvoice")
    assert support["arousal"] == "provisional_model_control"
    assert support["coarse_affect"] == "provisional_model_control"
