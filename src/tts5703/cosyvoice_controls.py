"""Pure CosyVoice3 acoustic-control mappings."""

_SEMANTIC_RATES = {
    "slow": 0.8,
    "normal": 1.0,
    "fast": 1.2,
}
_AROUSAL_INSTRUCTIONS = {
    "low": "Use a calm, soft, subdued delivery.",
    "medium": "Use a neutral, moderately expressive delivery.",
    "high": "Use an energetic, intense delivery.",
}
_AFFECT_INSTRUCTIONS = {
    "neutral": "Use a neutral, composed tone.",
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


def _check_mapping(field: str, value: str | None, mapping: dict[str, str]) -> None:
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
            _AROUSAL_INSTRUCTIONS.get(arousal),
            _AFFECT_INSTRUCTIONS.get(coarse_affect),
        )
        if instruction is not None
    ]
    if not controls:
        return None
    return f"{_INSTRUCTION_PREFIX} {' '.join(controls)}{_END_OF_PROMPT}"
