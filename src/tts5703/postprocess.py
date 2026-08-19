"""Stage 4: generate telephone-quality audio."""

from typing import Any

from pydub import AudioSegment


def apply_telephone_effect(audio: AudioSegment, config: dict[str, Any]) -> AudioSegment:
    telephone = config["telephone"]
    processed = audio.set_frame_rate(telephone["sample_rate"]).set_channels(
        telephone["channels"]
    )
    processed = processed.high_pass_filter(telephone["high_pass_hz"])
    processed = processed.low_pass_filter(telephone["low_pass_hz"])
    return processed - telephone["volume_db_reduction"]
