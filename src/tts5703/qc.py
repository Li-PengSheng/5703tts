"""Stage 6: basic output completeness checks."""

from dataclasses import dataclass, field
from pathlib import Path

from pydub import AudioSegment

from .validate import NormalizedDialogue

REQUIRED_TURN_FIELDS = {"turn_id", "speaker", "text", "label", "start_time", "end_time"}


@dataclass
class QCResult:
    dialogue_id: str
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


def run_qc(
    dialogue: NormalizedDialogue,
    out_dir: Path,
    clean_path: Path,
    telephone_path: Path,
    metadata: dict,
) -> QCResult:
    checks: dict[str, bool] = {}
    issues: list[str] = []
    turn_audio_ok = True
    metadata_by_turn_id = {turn["turn_id"]: turn for turn in metadata.get("turns", [])}
    for turn in dialogue.turns:
        metadata_turn = metadata_by_turn_id.get(turn.turn_id)
        if metadata_turn is None:
            turn_audio_ok = False
            issues.append(f"turn {turn.turn_id} 不在 metadata 中")
            continue
        path = out_dir / metadata_turn["turn_audio"]
        if not path.exists():
            turn_audio_ok = False
            issues.append(f"turn {turn.turn_id} 的音频文件缺失: {path.name}")
            continue
        try:
            if len(AudioSegment.from_file(path)) / 1000 < 0.1:
                turn_audio_ok = False
                issues.append(f"turn {turn.turn_id} 音频时长异常（应大于 0.1 秒）")
        except Exception as error:
            turn_audio_ok = False
            issues.append(f"turn {turn.turn_id} 音频无法读取: {error}")
    checks["turn_audio_files_exist"] = turn_audio_ok

    checks["clean_audio_exists"] = clean_path.exists()
    checks["telephone_audio_exists"] = telephone_path.exists()
    if not clean_path.exists():
        issues.append("clean 版音频未生成")
    if not telephone_path.exists():
        issues.append("telephone 版音频未生成")

    duration_ok = clean_path.exists()
    if duration_ok:
        duration = len(AudioSegment.from_file(clean_path)) / 1000
        turns = metadata.get("turns", [])
        if duration <= 0 or (turns and turns[-1]["end_time"] > duration + 0.05):
            duration_ok = False
            issues.append("clean 音频时长或最后一个 turn 的时间戳异常")
    checks["duration_positive"] = duration_ok

    turns = metadata.get("turns", [])
    checks["turn_count_matches"] = len(turns) == len(dialogue.turns)
    if not checks["turn_count_matches"]:
        issues.append("输入与 metadata 的 turn 数量不一致")

    previous_end = -1.0
    timestamps_ok = True
    for turn in turns:
        if turn["start_time"] < previous_end or turn["end_time"] <= turn["start_time"]:
            timestamps_ok = False
            issues.append(f"turn {turn['turn_id']} 时间戳异常")
        previous_end = turn["end_time"]
    checks["timestamps_increasing"] = timestamps_ok

    fields_ok = all(not (REQUIRED_TURN_FIELDS - set(turn)) for turn in turns)
    checks["metadata_fields_complete"] = fields_ok
    if not fields_ok:
        issues.append("metadata 缺少硬性契约字段")
    return QCResult(dialogue.dialogue_id, all(checks.values()), checks, issues)
