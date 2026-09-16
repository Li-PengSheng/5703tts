"""Stage 5: write the turn-label-audio alignment contract."""

import json
from copy import deepcopy
from pathlib import Path

from .assemble import TurnTiming
from .cosyvoice_controls import resolve_cosyvoice_controls
from .engine_capabilities import (
    control_support,
    has_declared_capabilities,
    ignored_requested_controls,
    requested_acoustic_spec,
)
from .tts_engine import resolve_higgs_turn_controls


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


def write_metadata(metadata: dict, out_dir: Path) -> Path:
    """Write the metadata record beside the dialogue audio outputs."""
    output_path = out_dir / f"{metadata['dialogue_id']}_metadata.json"
    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_path
