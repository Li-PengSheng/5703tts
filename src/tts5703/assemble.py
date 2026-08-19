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
    pause_after_ms: int
    emotion: str | None
    arousal: str | None
    paralinguistic_events: list[Any]
    start_sec: float
    end_sec: float


def assemble_dialogue(
    turns: list[NormalizedTurn],
    turn_audio_paths: dict[int, Path],
    config: dict[str, Any],
) -> tuple[AudioSegment, list[TurnTiming]]:
    # 注意: pydub 的 append(..., crossfade=N) 是"重叠式拼接"，
    # 会让新 segment 提前 N ms 开始与前一段混合，导致 len(audio) 记录的位置
    # 和真实波形里的分界点不一致，而且这个偏差会随 turn 数量累积增大。
    # 危机对话里 caller/counsellor 的话本身也不该被强行叠在一起，
    # 所以这里改成不叠(crossfade=0)，只用轻微的 fade in/out 去掉硬切的爆音感。
    fade_ms = config.get("fade_ms", 5)
    audio = AudioSegment.empty()
    timings: list[TurnTiming] = []

    for turn in turns:
        segment = (
            AudioSegment.from_file(turn_audio_paths[turn.turn_id])
            .fade_in(fade_ms)
            .fade_out(fade_ms)
        )
        start_ms = len(audio)
        audio = segment if not audio else audio + segment  # 直接首尾相接，不重叠
        end_ms = len(audio)
        timings.append(
            TurnTiming(
                turn_id=turn.turn_id,
                speaker=turn.speaker,
                text=turn.text,
                label=turn.label,
                rate=turn.rate,
                pause_after_ms=turn.pause_after_ms,
                emotion=turn.emotion,
                arousal=turn.arousal,
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
