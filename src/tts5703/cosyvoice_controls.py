"""Pure CosyVoice3 acoustic-control mappings."""

from typing import Any

COSYVOICE_CONTROL_MAPPING_NAME = "cosyvoice3_control_mapping"
COSYVOICE_CONTROL_MAPPING_VERSION = "v1"
COSYVOICE_CONTROL_MAPPING_STATUS = "provisional"

_SEMANTIC_RATES = {
    "slow": 0.8,
    "normal": 1.0,
    "fast": 1.2,
}
_AROUSAL_INSTRUCTIONS: dict[str, str | None] = {
    "low": (
        "Keep vocal activation clearly low, with subdued energy and restrained "
        "emphasis. Do not deliberately change the speaking rate."
    ),
    "medium": None,
    "high": (
        "Use clearly high vocal activation, with stronger energy, projection, and "
        "emphasis, and a more animated delivery. Do not deliberately change the "
        "speaking rate."
    ),
}
_AFFECT_INSTRUCTIONS: dict[str, str | None] = {
    "neutral": (
        "Speak naturally in a neutral, emotionally even conversational manner."
    ),
    "sad": "Speak in a clearly sad and downcast manner.",
    "anxious": (
        "Speak in a clearly anxious, worried, and uneasy manner, with noticeable "
        "nervous tension and uncertainty, but not panic or urgency."
    ),
    "angry": "Speak in a clearly angry, firm, and forceful manner, without shouting.",
    "warm": (
        "Speak in a clearly warm, gentle, compassionate, supportive, and reassuring "
        "manner, with an emotionally present and caring delivery rather than a "
        "cheerful or excited one."
    ),
    "distressed": "Use a distressed, worried, and sad tone.",
}
_INSTRUCTION_PREFIX = "You are a helpful assistant."
_END_OF_PROMPT = "<|endofprompt|>"


class BackendControlError(ValueError):
    """Raised when a schema-valid control has no CosyVoice mapping."""


def rate_to_cosyvoice_speed(rate: str) -> float:
    """Map semantic or legacy percentage rates to CosyVoice's speed argument."""
    if rate in _SEMANTIC_RATES:
        return _SEMANTIC_RATES[rate]
    return max(0.1, 1 + int(rate[:-1]) / 100)


def _check_mapping(
    field: str, value: str | None, mapping: dict[str, str | None]
) -> None:
    if value is None or value in mapping:
        return
    raise BackendControlError(
        f"Unsupported CosyVoice {field} mapping: {value!r}. "
        f"Currently supported mappings: {', '.join(mapping)}."
    )


def validate_cosyvoice_controls(arousal: str | None, coarse_affect: str | None) -> None:
    """Reject requested controls that have no current CosyVoice mapping."""
    _check_mapping("arousal", arousal, _AROUSAL_INSTRUCTIONS)
    _check_mapping("coarse_affect", coarse_affect, _AFFECT_INSTRUCTIONS)


def build_cosyvoice_instruction(
    arousal: str | None, coarse_affect: str | None
) -> str | None:
    """Build the current CosyVoice3 instruction from requested controls."""
    validate_cosyvoice_controls(arousal, coarse_affect)
    controls = [
        instruction
        for instruction in (
            _AFFECT_INSTRUCTIONS.get(coarse_affect),
            _AROUSAL_INSTRUCTIONS.get(arousal),
        )
        if instruction is not None
    ]
    if arousal is None and coarse_affect is None:
        return None
    return f"{_INSTRUCTION_PREFIX} {' '.join(controls)}{_END_OF_PROMPT}"


def resolve_cosyvoice_controls(
    rate: str, arousal: str | None, coarse_affect: str | None
) -> dict[str, Any]:
    """Describe the backend controls used for synthesis, without fidelity claims."""
    instruction = build_cosyvoice_instruction(arousal, coarse_affect)
    affect_resolution = {
        "requested": coarse_affect,
        "method": "instruction" if coarse_affect is not None else "not_requested",
        "mapping_value": coarse_affect,
    }
    if coarse_affect == "distressed":
        affect_resolution["compatibility"] = "legacy"
    return {
        "backend": "cosyvoice",
        "mapping": {
            "name": COSYVOICE_CONTROL_MAPPING_NAME,
            "version": COSYVOICE_CONTROL_MAPPING_VERSION,
            "status": COSYVOICE_CONTROL_MAPPING_STATUS,
        },
        "inference_mode": "instruct2" if instruction is not None else "zero_shot",
        "speaking_rate": {
            "requested": rate,
            "method": "numeric_speed",
            "speed": rate_to_cosyvoice_speed(rate),
        },
        "arousal": {
            "requested": arousal,
            "method": (
                "not_requested"
                if arousal is None
                else "no_additional_clause"
                if _AROUSAL_INSTRUCTIONS[arousal] is None
                else "instruction"
            ),
            "mapping_value": arousal,
        },
        "coarse_affect": affect_resolution,
        "resolved_instruction": instruction,
    }
