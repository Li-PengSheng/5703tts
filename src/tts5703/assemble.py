"""Stage 3: join turn audio and produce turn-level timestamps."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydub import AudioSegment

from .validate import NormalizedTurn


@dataclass
class TurnTiming:
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
