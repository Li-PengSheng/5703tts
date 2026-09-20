"""Production control capabilities for the two final-schema backends."""

from copy import deepcopy
from typing import Any

MODEL_CONTROL = "model_control"
PROVISIONAL_MODEL_CONTROL = "provisional_model_control"
PIPELINE_TIMING = "pipeline_timing"
PIPELINE_POSTPROCESS = "pipeline_postprocess"
UNSUPPORTED = "unsupported"

CAPABILITY_VOCABULARY: frozenset[str] = frozenset(
    {
        MODEL_CONTROL,
        PROVISIONAL_MODEL_CONTROL,
        PIPELINE_TIMING,
        PIPELINE_POSTPROCESS,
        UNSUPPORTED,
    }
)

# These declarations describe ownership/support only; an individual turn's
# realized plan remains the cached output of controlled_tts.map_turn_to_higgs.
CONTROLLED_TTS_V1_CAPABILITIES: dict[str, Any] = {
    "required": {
        "rate": {"support": PIPELINE_POSTPROCESS, "realization": "ffmpeg_atempo"},
        "arousal": {"support": MODEL_CONTROL, "realization": "higgs_model_control"},
        "affect": {"support": MODEL_CONTROL, "realization": "higgs_model_control"},
        "pause_before": {
            "support": PIPELINE_TIMING,
            "realization": "prepared_dialogue_assembly",
        },
        "pause_within": {
            "support": MODEL_CONTROL,
            "realization": "native_higgs_pause_token_planner",
        },
        "hesitations": {
            "support": MODEL_CONTROL,
            "realization": "lexical_planner",
        },
    },
    "best_effort": {
        "affect_fine": {
            "support": "conditional_joint_realization",
            "possible_statuses": [
                "jointly_realized",
                "not_realized",
                "unsupported",
            ],
        },
        "volume": {
            "support": UNSUPPORTED,
            "possible_statuses": ["not_realized", "unsupported"],
        },
        "flattened_affect": {
            "support": UNSUPPORTED,
            "possible_statuses": ["not_realized", "unsupported"],
        },
        "events": {
            "support": UNSUPPORTED,
            "possible_statuses": ["not_realized", "unsupported"],
        },
    },
}

COSYVOICE3_CAPABILITIES: dict[str, Any] = {
    "required": {
        "rate": {"support": MODEL_CONTROL, "realization": "numeric_speed"},
        "arousal": {
            "support": PROVISIONAL_MODEL_CONTROL,
            "realization": "instruction_mapping",
        },
        "affect": {
            "support": PROVISIONAL_MODEL_CONTROL,
            "realization": "instruction_mapping",
        },
        "pause_before": {
            "support": PIPELINE_TIMING,
            "realization": "prepared_dialogue_assembly",
        },
        "pause_within": {
            "support": UNSUPPORTED,
            "realization": "fail_closed_when_requested",
        },
        "hesitations": {
            "support": MODEL_CONTROL,
            "realization": "shared_lexical_planner",
        },
    },
    "best_effort": {
        field: {
            "support": UNSUPPORTED,
            "possible_statuses": ["not_realized", "unsupported"],
        }
        for field in ("affect_fine", "volume", "flattened_affect", "events")
    },
}


class UnknownEngineCapabilityError(KeyError):
    """Raised when no capability declaration exists for the requested engine."""


def controlled_tts_v1_capabilities(engine: str = "higgs") -> dict[str, Any]:
    """Return the selected backend's isolated final-schema capabilities."""
    if engine == "higgs":
        return deepcopy(CONTROLLED_TTS_V1_CAPABILITIES)
    if engine == "cosyvoice":
        return deepcopy(COSYVOICE3_CAPABILITIES)
    raise UnknownEngineCapabilityError(
        f"No final-schema capabilities are declared for engine {engine!r}"
    )
