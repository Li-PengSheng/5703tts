"""Assemble speech-only turn WAVs into a timed clean dialogue.

``pause_before`` belongs to assembly: silence is inserted before timestamp
``start_sec`` and never baked into a turn WAV. A short fade only protects hard
waveform edges; turns are concatenated without crossfade. Prepared-turn and
result ordinals/source IDs must match exactly before assembly is trusted.
"""

from typing import Any

from pydub import AudioSegment

from ..render_models import PreparedDialogue, TurnRenderResult


def assemble_prepared_dialogue(
    dialogue: PreparedDialogue,
    turn_results: tuple[TurnRenderResult, ...] | list[TurnRenderResult],
    config: dict[str, Any],
) -> tuple[AudioSegment, list[dict[str, Any]]]:
    """Join ordered speech with assembly-owned, unfaded pre-speech pauses."""
    by_ordinal: dict[int, TurnRenderResult] = {}
    for result in turn_results:
        if result.ordinal in by_ordinal:
            raise ValueError(
                f"Duplicate prepared turn result ordinal: {result.ordinal}"
            )
        by_ordinal[result.ordinal] = result
    ordered_turns = sorted(dialogue.turns, key=lambda turn: turn.ordinal)
    ordinals = [turn.ordinal for turn in ordered_turns]
    if ordinals != list(range(1, len(ordered_turns) + 1)):
        raise ValueError("Prepared dialogue ordinals must be unique and 1-based")
    expected = set(ordinals)
    if set(by_ordinal) != expected:
        raise ValueError("Prepared turn results do not match dialogue ordinals")

    fade_ms = config.get("fade_ms", 5)
    audio = AudioSegment.empty()
    timings: list[dict[str, Any]] = []
    for turn in ordered_turns:
        result = by_ordinal[turn.ordinal]
        if result.source_turn_id != turn.source_turn_id:
            raise ValueError(
                f"Prepared result source_turn_id mismatch at ordinal {turn.ordinal}"
            )
        segment = AudioSegment.from_file(result.output_path)
        if turn.pause_before_ms:
            pause = (
                AudioSegment.silent(
                    duration=turn.pause_before_ms,
                    frame_rate=segment.frame_rate,
                )
                .set_sample_width(segment.sample_width)
                .set_channels(segment.channels)
            )
            audio = pause if not audio else audio + pause
        segment = segment.fade_in(fade_ms).fade_out(fade_ms)
        start_ms = len(audio)
        audio = segment if not audio else audio + segment
        end_ms = len(audio)
        timings.append(
            {
                "ordinal": turn.ordinal,
                "source_turn_id": turn.source_turn_id,
                "pause_before_ms": turn.pause_before_ms,
                "pause_after_ms": 0,
                "start_sec": round(start_ms / 1000, 3),
                "end_sec": round(end_ms / 1000, 3),
            }
        )
    return audio, timings
