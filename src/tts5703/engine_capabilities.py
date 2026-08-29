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
``unsupported``
    The backend cannot consume this control; a requested value is ignored.
"""

from collections.abc import Mapping
from typing import Any

MODEL_CONTROL = "model_control"
PROVISIONAL_MODEL_CONTROL = "provisional_model_control"
PIPELINE_TIMING = "pipeline_timing"
UNSUPPORTED = "unsupported"

CAPABILITY_VOCABULARY: frozenset[str] = frozenset(
    {MODEL_CONTROL, PROVISIONAL_MODEL_CONTROL, PIPELINE_TIMING, UNSUPPORTED}
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
