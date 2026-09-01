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
    """Check basic output completeness and turn-timestamp consistency.

    This is workstream-level structural QC, not perceptual speech evaluation or
    formal downstream crisis-detection validation. The caller logs/returns this
    result; no separate QC report file is written by the current pipeline.
    """
    checks: dict[str, bool] = {}
    issues: list[str] = []
    turn_audio_ok = True
    metadata_by_turn_id = {turn["turn_id"]: turn for turn in metadata.get("turns", [])}
    for turn in dialogue.turns:
        metadata_turn = metadata_by_turn_id.get(turn.turn_id)
        if metadata_turn is None:
            turn_audio_ok = False
            issues.append(f"turn {turn.turn_id} is missing from metadata")
            continue
        path = out_dir / metadata_turn["turn_audio"]
        if not path.exists():
            turn_audio_ok = False
            issues.append(f"turn {turn.turn_id} audio file is missing: {path.name}")
            continue
        try:
            if len(AudioSegment.from_file(path)) / 1000 < 0.1:
                turn_audio_ok = False
                issues.append(
                    f"turn {turn.turn_id} audio duration is invalid (must exceed 0.1 seconds)"
                )
        except Exception as error:  # noqa: BLE001 - report any decoder/backend failure as QC
            turn_audio_ok = False
            issues.append(f"turn {turn.turn_id} audio cannot be read: {error}")
    checks["turn_audio_files_exist"] = turn_audio_ok

    checks["clean_audio_exists"] = clean_path.exists()
    checks["telephone_audio_exists"] = telephone_path.exists()
    if not clean_path.exists():
        issues.append("Clean audio was not generated")
    if not telephone_path.exists():
        issues.append("Telephone audio was not generated")

    duration_ok = clean_path.exists()
    if duration_ok:
        duration = len(AudioSegment.from_file(clean_path)) / 1000
        turns = metadata.get("turns", [])
        if duration <= 0 or (turns and turns[-1]["end_time"] > duration + 0.05):
            duration_ok = False
            issues.append("Clean audio duration or final-turn timestamp is invalid")
    checks["duration_positive"] = duration_ok

    turns = metadata.get("turns", [])
    checks["turn_count_matches"] = len(turns) == len(dialogue.turns)
    if not checks["turn_count_matches"]:
        issues.append("Input and metadata turn counts do not match")

    previous_end = -1.0
    timestamps_ok = True
    for turn in turns:
        if turn["start_time"] < previous_end or turn["end_time"] <= turn["start_time"]:
            timestamps_ok = False
            issues.append(f"turn {turn['turn_id']} timestamps are invalid")
        previous_end = turn["end_time"]
    checks["timestamps_increasing"] = timestamps_ok

    fields_ok = all(not (REQUIRED_TURN_FIELDS - set(turn)) for turn in turns)
    checks["metadata_fields_complete"] = fields_ok
    if not fields_ok:
        issues.append("Metadata is missing required contract fields")
    return QCResult(dialogue.dialogue_id, all(checks.values()), checks, issues)
