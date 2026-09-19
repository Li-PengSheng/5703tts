"""Production backend capability declarations."""

import pytest

from tts5703.backends import capabilities


@pytest.mark.parametrize("engine", ["higgs", "cosyvoice"])
def test_capability_values_use_the_declared_vocabulary(engine: str) -> None:
    declaration = capabilities.controlled_tts_v1_capabilities(engine)

    for section in declaration.values():
        for control in section.values():
            support = control["support"]
            if support != "conditional_joint_realization":
                assert support in capabilities.CAPABILITY_VOCABULARY


def test_higgs_and_cosyvoice_declare_truthful_pause_within_support() -> None:
    higgs = capabilities.controlled_tts_v1_capabilities("higgs")
    cosyvoice = capabilities.controlled_tts_v1_capabilities("cosyvoice")

    assert higgs["required"]["pause_within"] == {
        "support": "model_control",
        "realization": "native_higgs_pause_token_planner",
    }
    assert cosyvoice["required"]["pause_within"] == {
        "support": "unsupported",
        "realization": "fail_closed_when_requested",
    }


def test_capability_results_are_isolated_copies() -> None:
    result = capabilities.controlled_tts_v1_capabilities("higgs")
    result["required"]["rate"]["support"] = "unsupported"

    assert (
        capabilities.controlled_tts_v1_capabilities("higgs")["required"]["rate"][
            "support"
        ]
        == "pipeline_postprocess"
    )


def test_unknown_engine_is_rejected() -> None:
    with pytest.raises(capabilities.UnknownEngineCapabilityError):
        capabilities.controlled_tts_v1_capabilities("unknown")
