"""Single source of truth for per-backend acoustic-control capabilities.

Canonical dialogue input (schema v0.2) records *requested* acoustic intent and
stays backend-independent. Each backend consumes only part of that intent, so
both production metadata and the controlled benchmark need one shared answer to
"what does this engine actually do with this control?".

Support vocabulary:

``model_control``
    The backend receives this control and it is expected to change acoustics.
``provisional_model_control``
    The backend receives this control through a mapping that has not yet been
    validated for acoustic fidelity; treat it as unverified, not as evidence.
``pipeline_timing``
    Not sent to the backend at all; realised by this project's own assembly
    stage, so it is honoured regardless of backend support.
``pipeline_postprocess``
    Not sent to the backend as a model control; planned for this project's
    audio postprocessing stage.
``unsupported``
    The backend cannot consume this control; a requested value is ignored.
"""

from collections.abc import Mapping
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

# Declaration order is also the reporting order of derived control lists.
ACOUSTIC_CONTROL_FIELDS: tuple[str, ...] = (
    "rate",
    "pause_before_ms",
    "pause_after_ms",
    "arousal",
    "coarse_affect",
    "emotion",
    "paralinguistic_events",
)

ENGINE_CAPABILITIES: dict[str, dict[str, dict[str, str]]] = {
    "higgs": {
        "rate": {"support": PIPELINE_POSTPROCESS},
        "pause_before_ms": {"support": PIPELINE_TIMING},
        "pause_after_ms": {"support": PIPELINE_TIMING},
        "arousal": {"support": PROVISIONAL_MODEL_CONTROL},
        "coarse_affect": {"support": PROVISIONAL_MODEL_CONTROL},
        "emotion": {"support": UNSUPPORTED},
        "paralinguistic_events": {"support": UNSUPPORTED},
    },
    "cosyvoice": {
        "rate": {"support": MODEL_CONTROL},
        "pause_before_ms": {"support": PIPELINE_TIMING},
        "pause_after_ms": {"support": PIPELINE_TIMING},
        "arousal": {"support": PROVISIONAL_MODEL_CONTROL},
        "coarse_affect": {"support": PROVISIONAL_MODEL_CONTROL},
        "emotion": {"support": UNSUPPORTED},
        "paralinguistic_events": {"support": UNSUPPORTED},
    },
    "kokoro": {
        "rate": {"support": MODEL_CONTROL},
        "pause_before_ms": {"support": PIPELINE_TIMING},
        "pause_after_ms": {"support": PIPELINE_TIMING},
        "arousal": {"support": UNSUPPORTED},
        "coarse_affect": {"support": UNSUPPORTED},
        "emotion": {"support": UNSUPPORTED},
        "paralinguistic_events": {"support": UNSUPPORTED},
    },
}

# Final-schema controls intentionally live outside the v0.2 field vocabulary.
# These declarations describe ownership/support only; an individual turn's
# realized plan remains the cached output of controlled_tts.map_turn_to_higgs.
FINAL_CONTROLLED_TTS_V1_CAPABILITIES: dict[str, Any] = {
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

FINAL_COSYVOICE3_CAPABILITIES: dict[str, Any] = {
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


def declared_engines() -> list[str]:
    return sorted(ENGINE_CAPABILITIES)


def has_declared_capabilities(engine: str | None) -> bool:
    return engine in ENGINE_CAPABILITIES


def engine_capabilities(engine: str) -> dict[str, dict[str, str]]:
    """Return an isolated copy of one engine's declared capability map."""
    try:
        capabilities = ENGINE_CAPABILITIES[engine]
    except KeyError as error:
        raise UnknownEngineCapabilityError(
            f"No acoustic-control capabilities are declared for engine {engine!r}; "
            f"declared engines: {declared_engines()}"
        ) from error
    return {field: dict(description) for field, description in capabilities.items()}


def control_support(engine: str) -> dict[str, str]:
    """Return a flat ``field -> support`` map for one engine."""
    return {
        field: description["support"]
        for field, description in engine_capabilities(engine).items()
    }


def final_controlled_tts_v1_capabilities(engine: str = "higgs") -> dict[str, Any]:
    """Return the selected backend's isolated final-schema capabilities."""
    if engine == "higgs":
        return deepcopy(FINAL_CONTROLLED_TTS_V1_CAPABILITIES)
    if engine == "cosyvoice":
        return deepcopy(FINAL_COSYVOICE3_CAPABILITIES)
    raise UnknownEngineCapabilityError(
        f"No final-schema capabilities are declared for engine {engine!r}"
    )


def requested_acoustic_spec(turn: Any) -> dict[str, Any]:
    """Snapshot the requested acoustic controls of a normalized turn or timing."""
    return {field: getattr(turn, field) for field in ACOUSTIC_CONTROL_FIELDS}


def _is_requested(field: str, value: Any) -> bool:
    """Treat null and empty event lists as "not requested" rather than ignored."""
    if value is None:
        return False
    if field == "paralinguistic_events":
        return bool(value)
    return True


def ignored_requested_controls(
    engine: str, requested_spec: Mapping[str, Any]
) -> list[str]:
    """List requested controls the engine declares it cannot consume.

    Pipeline timing controls are never listed: they are honoured by this
    project's assembly stage rather than by the backend.
    """
    support = control_support(engine)
    return [
        field
        for field in ACOUSTIC_CONTROL_FIELDS
        if field in requested_spec
        and _is_requested(field, requested_spec[field])
        and support[field] == UNSUPPORTED
    ]
