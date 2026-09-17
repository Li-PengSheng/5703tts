"""Stage 3: join turn audio and produce turn-level timestamps."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydub import AudioSegment

from .render_plan import PreparedDialogue, TurnRenderResult
from .validate import NormalizedTurn


@dataclass
class TurnTiming:
    """Requested turn metadata plus speech-only boundaries in assembled audio.

    The boundaries exclude the turn's pauses: ``pause_before_ms`` advances
    ``start_sec``, while ``pause_after_ms`` is appended after ``end_sec``.
    """

    turn_id: int
    speaker: str
    text: str
    label: str
    rate: str
    pause_before_ms: int
    pause_after_ms: int
    emotion: str | None
    arousal: str | None
    coarse_affect: str | None
    paralinguistic_events: list[Any]
    start_sec: float
    end_sec: float


def assemble_dialogue(
    turns: list[NormalizedTurn],
    turn_audio_paths: dict[int, Path],
    config: dict[str, Any],
) -> tuple[AudioSegment, list[TurnTiming]]:
    """Concatenate turn files and realise pauses as deterministic timeline gaps.

    Returns the clean dialogue waveform in memory and one alignment record per
    input turn. Per-turn files are read in the validated turn order; dictionary
    ordering of ``turn_audio_paths`` is irrelevant.
    """
    # pydub's append(..., crossfade=N) overlaps segments. The next segment starts
    # N ms early, so len(audio) no longer matches the real waveform boundary and
    # the discrepancy accumulates with every turn. Dialogue turns should not
    # overlap, so use direct joins and short fades to avoid hard-cut clicks.
    fade_ms = config.get("fade_ms", 5)
    audio = AudioSegment.empty()
    timings: list[TurnTiming] = []

    for turn in turns:
        if turn.pause_before_ms > 0:
            pause_before = (
                AudioSegment.silent(duration=turn.pause_before_ms)
                .fade_in(fade_ms)
                .fade_out(fade_ms)
            )
            audio = audio + pause_before
        segment = (
            AudioSegment.from_file(turn_audio_paths[turn.turn_id])
            .fade_in(fade_ms)
            .fade_out(fade_ms)
        )
        start_ms = len(audio)
        audio = segment if not audio else audio + segment  # Direct join; no overlap.
        end_ms = len(audio)
        timings.append(
            TurnTiming(
                turn_id=turn.turn_id,
                speaker=turn.speaker,
                text=turn.text,
                label=turn.label,
                rate=turn.rate,
                pause_before_ms=turn.pause_before_ms,
                pause_after_ms=turn.pause_after_ms,
                emotion=turn.emotion,
                arousal=turn.arousal,
                coarse_affect=turn.coarse_affect,
                paralinguistic_events=turn.paralinguistic_events,
                start_sec=round(start_ms / 1000, 3),
                end_sec=round(end_ms / 1000, 3),
            )
        )
        pause_ms = turn.pause_after_ms
        if pause_ms > 0:
            pause = (
                AudioSegment.silent(duration=pause_ms)
                .fade_in(fade_ms)
                .fade_out(fade_ms)
            )
            audio = audio + pause
    return audio, timings


def assemble_prepared_dialogue(
    dialogue: PreparedDialogue,
    turn_results: tuple[TurnRenderResult, ...] | list[TurnRenderResult],
    config: dict[str, Any],
) -> tuple[AudioSegment, list[dict[str, Any]]]:
    """Assemble final speech with only unfaded pre-speech pauses."""
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
