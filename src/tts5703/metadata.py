"""Stage 5: write the turn-label-audio alignment contract."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .assemble import TurnTiming
from .cosyvoice_controls import resolve_cosyvoice_controls
from .engine_capabilities import (
    control_support,
    final_controlled_tts_v1_capabilities,
    has_declared_capabilities,
    ignored_requested_controls,
    requested_acoustic_spec,
)
from .higgs_controls import resolve_higgs_turn_controls
from .input_records import InputRecord
from .render_plan import PreparedDialogue, TurnRenderResult


def _control_resolution(timing: TurnTiming, engine: str | None) -> dict | None:
    if engine == "cosyvoice":
        return resolve_cosyvoice_controls(
            timing.rate, timing.arousal, timing.coarse_affect
        )
    if engine == "higgs":
        # TurnTiming preserves every resolver input from the normalized turn. The
        # deterministic helper is shared with synthesis so metadata cannot drift
        # to a separately implemented mapping.
        resolution = deepcopy(resolve_higgs_turn_controls(timing))
        rate = resolution["postprocess"]["rate"]
        execution_status = "executed" if rate["enabled"] else "not_required"
        rate["execution_status"] = execution_status
        resolution["realization"]["speaking_rate"]["execution_status"] = (
            execution_status
        )
        resolution["caveats"] = [
            caveat
            for caveat in resolution["caveats"]
            if not ("rate postprocessing" in caveat and "not been executed" in caveat)
        ]
        return resolution
    return None


def _higgs_speaker_reference(timing: TurnTiming, engine_info: dict) -> dict:
    reference = engine_info["references"][timing.speaker]
    return {
        "speaker_id": timing.speaker,
        "reference_wav": reference["reference_wav"],
        "sha256": reference["sha256"],
    }


def build_metadata(
    dialogue_id: str,
    clean_path: Path,
    telephone_path: Path,
    timings: list[TurnTiming],
    turn_audio_paths: dict[int, Path],
    engine_info: dict,
) -> dict:
    """Build the per-dialogue alignment and requested-control provenance record.

    Values copied from ``TurnTiming`` describe requested intent and pipeline
    timing, not measurements made from the generated waveform. ``engine_info``
    likewise contains configuration declarations unless its key explicitly says
    that a value was runtime-observed.
    """
    # Keep defensive null handling for callers that pass an unknown engine, so
    # "nothing declared" cannot be read as "nothing ignored".
    engine = engine_info.get("engine")
    declared = has_declared_capabilities(engine)
    support = control_support(engine) if declared else None
    return {
        "dialogue_id": dialogue_id,
        "clean_audio": clean_path.name,
        "telephone_audio": telephone_path.name,
        # Record the engine and configuration used for this render so datasets
        # remain traceable during future engine-comparison experiments.
        "tts": {**engine_info, "control_support": support},
        "turns": [
            {
                "turn_id": timing.turn_id,
                "speaker": timing.speaker,
                "text": timing.text,
                "label": timing.label,
                "turn_audio": turn_audio_paths[timing.turn_id].name,
                # Flat requested fields are kept for backward compatibility and
                # duplicate requested_acoustic_spec exactly.
                "rate": timing.rate,
                "pause_before_ms": timing.pause_before_ms,
                "pause_after_ms": timing.pause_after_ms,
                "emotion": timing.emotion,
                "arousal": timing.arousal,
                "coarse_affect": timing.coarse_affect,
                "paralinguistic_events": timing.paralinguistic_events,
                "requested_acoustic_spec": requested_acoustic_spec(timing),
                "ignored_requested_controls": (
                    ignored_requested_controls(engine, requested_acoustic_spec(timing))
                    if declared
                    else None
                ),
                **(
                    {"control_resolution": _control_resolution(timing, engine)}
                    if engine in {"cosyvoice", "higgs"}
                    else {}
                ),
                **(
                    {"speaker_reference": _higgs_speaker_reference(timing, engine_info)}
                    if engine == "higgs"
                    else {}
                ),
                "start_time": timing.start_sec,
                "end_time": timing.end_sec,
            }
            for timing in timings
        ],
    }


def build_final_metadata(
    input_record: InputRecord,
    dialogue: PreparedDialogue,
    clean_path: Path,
    telephone_path: Path,
    timings: list[dict[str, Any]],
    turn_results: tuple[TurnRenderResult, ...] | list[TurnRenderResult],
    engine_info: dict[str, Any],
) -> dict[str, Any]:
    """Build final metadata only from source, cached plans, and execution results."""
    raw = input_record.raw
    if raw.get("dialogue_id") != dialogue.dialogue_id:
        raise ValueError("InputRecord and PreparedDialogue dialogue_id mismatch")
    if input_record.record_sha256 != dialogue.record_sha256:
        raise ValueError("InputRecord and PreparedDialogue record SHA mismatch")
    if not (
        len(raw.get("turns", []))
        == len(dialogue.turns)
        == len(timings)
        == len(turn_results)
    ):
        raise ValueError("Final metadata inputs have inconsistent turn counts")

    results = {result.ordinal: result for result in turn_results}
    timing_by_ordinal = {timing.get("ordinal"): timing for timing in timings}
    expected_ordinals = {turn.ordinal for turn in dialogue.turns}
    if set(results) != expected_ordinals or set(timing_by_ordinal) != expected_ordinals:
        raise ValueError("Final metadata inputs have inconsistent turn ordinals")

    mapping = engine_info["control_mapping"]
    mapping_provenance = mapping["provenance"]
    turns: list[dict[str, Any]] = []
    for source, prepared in zip(raw["turns"], dialogue.turns, strict=True):
        result = results[prepared.ordinal]
        timing = timing_by_ordinal[prepared.ordinal]
        if (
            source.get("turn_id") != prepared.source_turn_id
            or result.source_turn_id != prepared.source_turn_id
            or timing.get("source_turn_id") != prepared.source_turn_id
        ):
            raise ValueError(
                f"Final metadata turn identity mismatch at ordinal {prepared.ordinal}"
            )
        acoustic = source["acoustic"]
        turns.append(
            {
                "source_identity": {
                    "ordinal": prepared.ordinal,
                    "source_turn_id": prepared.source_turn_id,
                    "upstream_role": prepared.upstream_role,
                    "logical_role": prepared.logical_role,
                    "upstream_scenario_speaker_id": (
                        prepared.upstream_scenario_speaker_id
                    ),
                    "render_speaker_id": prepared.render_speaker_id,
                },
                "labels": deepcopy(source["labels"]),
                "requested": {
                    "acoustic": {
                        "required": deepcopy(acoustic["required"]),
                        "best_effort": deepcopy(acoustic.get("best_effort")),
                    }
                },
                "planned": prepared.plan,
                "approved_speaker_reference": {
                    "render_speaker_id": prepared.render_speaker_id,
                    "reference_wav": prepared.reference_wav,
                    "resolved_reference_wav": str(prepared.resolved_reference_wav),
                    "sha256": prepared.reference_sha256,
                },
                "execution": {
                    "synthesis_status": result.synthesis_status,
                    "rate_status": result.rate_status,
                    "turn_audio": result.output_path.name,
                },
                "timing": deepcopy(timing),
            }
        )

    return {
        "schema_family": "final_nested",
        "dialogue_id": dialogue.dialogue_id,
        "clean_audio": clean_path.name,
        "telephone_audio": telephone_path.name,
        "tts": {
            **deepcopy(engine_info),
            "control_support": final_controlled_tts_v1_capabilities(),
        },
        "provenance": {
            "record_sha256": dialogue.record_sha256,
            "assignment_sha256": dialogue.assignment_sha256,
            "registry_sha256": dialogue.registry_sha256,
            "active_speakers_sha256": dialogue.active_speakers_sha256,
            "mapping_version": mapping["mapping_version"],
            "release_status": mapping["release_status"],
            "contract_sha256": mapping_provenance["contract_sha256"],
            "implementation_id": mapping["implementation_id"],
        },
        "turns": turns,
    }


def write_metadata(metadata: dict, out_dir: Path) -> Path:
    """Write the metadata record beside the dialogue audio outputs."""
    output_path = out_dir / f"{metadata['dialogue_id']}_metadata.json"
    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_path
