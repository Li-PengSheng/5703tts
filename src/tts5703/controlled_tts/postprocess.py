"""CPU-only postprocess plan construction for Controlled TTS v1."""

from typing import Any


def build_postprocess_plan(
    rate_value: dict[str, Any], pause_value: dict[str, Any]
) -> dict[str, Any]:
    """Return instructions only; production audio execution is intentionally separate."""
    factor = rate_value["atempo_factor"]
    milliseconds = pause_value["milliseconds"]
    return {
        "order": ["ffmpeg_atempo", "prepend_pause_before"],
        "atempo_factor": factor,
        "pause_before_ms": milliseconds,
        "atempo": {
            "enabled": factor is not None,
            "factor": factor,
            "processor": "ffmpeg_atempo",
        },
        "pause_before": {
            "enabled": milliseconds > 0,
            "duration_ms": milliseconds,
            "sample_rate_hz": "actual_final_audio_sample_rate",
            "sample_count_formula": ("(duration_ms * sample_rate_hz + 500) // 1000"),
        },
    }
