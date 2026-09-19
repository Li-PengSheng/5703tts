"""Stage 5: write the turn-label-audio alignment contract."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .backends.capabilities import controlled_tts_v1_capabilities
from .input.records import InputRecord
from .render_models import PreparedDialogue, TurnRenderResult


def build_metadata(
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

    engine = engine_info["engine"]
    mapping = engine_info["control_mapping"]
    mapping_provenance = (
        {
            "mapping_version": mapping["mapping_version"],
            "release_status": mapping["release_status"],
            "contract_sha256": mapping["provenance"]["contract_sha256"],
            "implementation_id": mapping["implementation_id"],
        }
        if engine == "higgs"
        else {
            "mapping_name": mapping["name"],
            "mapping_version": mapping["version"],
            "release_status": mapping["status"],
            "implementation_id": mapping["implementation_id"],
        }
    )
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
                "approved_speaker_reference": prepared.approved_reference,
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
            "control_support": controlled_tts_v1_capabilities(engine),
        },
        "provenance": {
            "record_sha256": dialogue.record_sha256,
            "assignment_sha256": dialogue.assignment_sha256,
            "registry_sha256": dialogue.registry_sha256,
            "active_speakers_sha256": dialogue.active_speakers_sha256,
            **mapping_provenance,
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
