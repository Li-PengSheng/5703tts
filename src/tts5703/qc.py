"""Stage 6: basic output completeness checks."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydub import AudioSegment

from .input.records import InputRecord
from .render_models import PreparedDialogue, TurnRenderResult


@dataclass
class QCResult:
    dialogue_id: str
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


def run_qc(
    input_record: InputRecord,
    dialogue: PreparedDialogue,
    turn_results: tuple[TurnRenderResult, ...] | list[TurnRenderResult],
    timings: list[dict[str, Any]],
    clean_path: Path,
    telephone_path: Path,
    metadata: dict[str, Any],
) -> QCResult:
    """Validate final structural/control integrity; no acoustic quality is judged."""
    checks: dict[str, bool] = {"scope_structural_control_integrity_only": True}
    issues: list[str] = []

    def check(name: str, validator: Callable[[], bool], issue: str) -> None:
        try:
            passed = bool(validator())
        except Exception:  # noqa: BLE001 - QC reports decoder/shape failures
            passed = False
        checks[name] = passed
        if not passed:
            issues.append(issue)

    raw = input_record.raw
    source_turns = raw.get("turns", [])
    metadata_turns = metadata.get("turns", [])
    results = {result.ordinal: result for result in turn_results}
    timing_by_ordinal = {timing.get("ordinal"): timing for timing in timings}
    expected_rate = {"slow": 0.85, "normal": None, "fast": 1.15}
    expected_pause = {"none": 0, "short": 500, "long": 900}

    check(
        "dialogue_identity",
        lambda: (
            input_record.dialogue_id
            == dialogue.dialogue_id
            == metadata.get("dialogue_id")
            and input_record.record_sha256 == dialogue.record_sha256
            and metadata["provenance"]["record_sha256"] == dialogue.record_sha256
        ),
        "Dialogue identity or record SHA is inconsistent",
    )
    check(
        "turn_count_and_order",
        lambda: (
            len(source_turns)
            == len(dialogue.turns)
            == len(metadata_turns)
            == len(turn_results)
            == len(timings)
            and [turn.ordinal for turn in dialogue.turns]
            == list(range(1, len(dialogue.turns) + 1))
            and [turn["source_identity"]["ordinal"] for turn in metadata_turns]
            == [turn.ordinal for turn in dialogue.turns]
        ),
        "Final turn count or ordinal order is inconsistent",
    )

    source_ok = True
    plan_ok = True
    speaker_ok = True
    execution_ok = True
    timing_ok = True
    turn_audio_ok = True
    previous_end = 0.0
    for index, prepared in enumerate(dialogue.turns):
        try:
            source = source_turns[index]
            metadata_turn = metadata_turns[index]
            identity = metadata_turn["source_identity"]
            requested = metadata_turn["requested"]["acoustic"]
            source_ok &= (
                source["turn_id"] == prepared.source_turn_id
                and source["speaker"] == prepared.upstream_role
                and identity
                == {
                    "ordinal": prepared.ordinal,
                    "source_turn_id": prepared.source_turn_id,
                    "upstream_role": prepared.upstream_role,
                    "logical_role": prepared.logical_role,
                    "upstream_scenario_speaker_id": (
                        prepared.upstream_scenario_speaker_id
                    ),
                    "render_speaker_id": prepared.render_speaker_id,
                }
                and metadata_turn["labels"] == source["labels"]
                and requested["required"] == source["acoustic"]["required"]
                and requested["best_effort"] == source["acoustic"].get("best_effort")
            )

            plan = prepared.plan
            metadata_plan = metadata_turn["planned"]
            normalized = plan["normalized"]
            realization = plan["realization"]
            hesitation = plan["planner"]["hesitation"]
            pause_ms = expected_pause[normalized["pause_before"]]
            source_required = source["acoustic"]["required"]
            plan_ok &= (
                metadata_plan == plan
                and plan["requested"]["required"] == source_required
                and plan["requested"]["best_effort"]
                == source["acoustic"].get("best_effort", {})
                and normalized["rate"] == source_required["rate"]
                and normalized["pause_within_count"] == source_required["pause_within"]
                and normalized["hesitation_count"] == source_required["hesitations"]
                and hesitation["inserted_count"]
                == normalized["hesitation_count"]
                == realization["hesitation_count"]["inserted_events"]
                and plan["postprocess"]["pause_before_ms"] == pause_ms
                and realization["pause_before"]["milliseconds"] == pause_ms
            )
            if dialogue.engine == "higgs":
                higgs = plan["higgs"]
                within = plan["planner"]["pause_within"]
                rate_factor = expected_rate[normalized["rate"]]
                plan_ok &= (
                    higgs["model_input"]
                    == "".join(higgs["prefix_tokens"]) + higgs["text"]
                    and higgs["synthesis_call_count"] == 1
                    and within["inserted_count"]
                    == normalized["pause_within_count"]
                    == higgs["native_pause_token_count"]
                    == realization["pause_within_count"]["inserted_native_pause_tokens"]
                    and plan["postprocess"]["atempo_factor"] == rate_factor
                    and plan["postprocess"]["atempo"]["factor"] == rate_factor
                    and plan["postprocess"]["atempo"]["enabled"]
                    is (rate_factor is not None)
                    and "cosyvoice" not in plan
                )
            else:
                cosy = plan["cosyvoice"]
                plan_ok &= (
                    plan["backend"] == "cosyvoice"
                    and "higgs" not in plan
                    and cosy["speed"]
                    == {"slow": 0.8, "normal": 1.0, "fast": 1.2}[normalized["rate"]]
                    and cosy["prompt_wav"] == prepared.prompt_wav
                    and cosy["prompt_text"] == prepared.prompt_text
                    and cosy["reference_sha256"] == prepared.reference_sha256
                    and realization["pause_within_count"]["status"] == "not_required"
                )

            reference = metadata_turn["approved_speaker_reference"]
            speaker_ok &= reference == prepared.approved_reference and (
                prepared.upstream_scenario_speaker_id != prepared.render_speaker_id
            )

            result = results[prepared.ordinal]
            execution = metadata_turn["execution"]
            expected_rate_status = (
                "not_required"
                if dialogue.engine == "higgs" and prepared.rate == "normal"
                else "executed"
            )
            execution_ok &= (
                result.source_turn_id == prepared.source_turn_id
                and result.synthesis_status == "synthesized"
                and result.rate_status == expected_rate_status
                and execution["synthesis_status"] == result.synthesis_status
                and execution["rate_status"] == result.rate_status
                and execution["turn_audio"] == result.output_path.name
            )

            timing = timing_by_ordinal[prepared.ordinal]
            metadata_timing = metadata_turn["timing"]
            expected_start = previous_end + prepared.pause_before_ms / 1000
            timing_ok &= (
                metadata_timing == timing
                and timing["source_turn_id"] == prepared.source_turn_id
                and timing["pause_before_ms"] == prepared.pause_before_ms
                and timing["pause_after_ms"] == 0
                and abs(timing["start_sec"] - expected_start) <= 0.002
                and timing["end_sec"] > timing["start_sec"]
            )
            previous_end = timing["end_sec"]

            speech = AudioSegment.from_file(result.output_path)
            speech_seconds = len(speech) / 1000
            turn_audio_ok &= (
                speech_seconds > 0
                and abs((timing["end_sec"] - timing["start_sec"]) - speech_seconds)
                <= 0.05
            )
        except Exception:  # noqa: BLE001 - QC reports decoder/shape failures
            source_ok = plan_ok = speaker_ok = execution_ok = False
            timing_ok = turn_audio_ok = False

    checks["input_metadata_consistency"] = source_ok
    checks["cached_plan_consistency"] = plan_ok
    checks["speaker_reference_consistency"] = speaker_ok
    checks["execution_consistency"] = execution_ok
    checks["speech_timestamps_consistent"] = timing_ok
    checks["turn_audio_decodes"] = turn_audio_ok
    for name, passed, issue in (
        ("input_metadata_consistency", source_ok, "Input metadata is inconsistent"),
        ("cached_plan_consistency", plan_ok, "Cached control plan is inconsistent"),
        (
            "speaker_reference_consistency",
            speaker_ok,
            "Speaker reference identity is inconsistent",
        ),
        ("execution_consistency", execution_ok, "Execution status is inconsistent"),
        (
            "speech_timestamps_consistent",
            timing_ok,
            "Speech-only timestamps are inconsistent",
        ),
        ("turn_audio_decodes", turn_audio_ok, "Turn audio is missing or unreadable"),
    ):
        if not passed:
            issues.append(issue)

    def output_decodes(path: Path) -> bool:
        return path.is_file() and len(AudioSegment.from_file(path)) > 0

    check(
        "clean_audio_decodes",
        lambda: output_decodes(clean_path),
        "Clean dialogue audio is missing or unreadable",
    )
    check(
        "telephone_audio_decodes",
        lambda: output_decodes(telephone_path),
        "Telephone dialogue audio is missing or unreadable",
    )
    return QCResult(dialogue.dialogue_id, all(checks.values()), checks, issues)
