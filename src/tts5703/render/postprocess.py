"""Derive the limited telephone-labelled signal-processing output.

The effect is only mono conversion, resampling, high-pass/low-pass filtering,
and level reduction. It is not a telephony codec, packet-loss model, line-noise
generator, room simulation, or complete PSTN network simulation.
"""

from typing import Any

from pydub import AudioSegment


def apply_telephone_effect(audio: AudioSegment, config: dict[str, Any]) -> AudioSegment:
    """Derive a mono, resampled, band-limited copy from clean dialogue audio.

    This is signal processing only: it does not simulate a telephone codec,
    packet loss, line noise, or room/channel impulse responses.
    """
    telephone = config["telephone"]
    processed = audio.set_frame_rate(telephone["sample_rate"]).set_channels(
        telephone["channels"]
    )
    processed = processed.high_pass_filter(telephone["high_pass_hz"])
    processed = processed.low_pass_filter(telephone["low_pass_hz"])
    return processed - telephone["volume_db_reduction"]
