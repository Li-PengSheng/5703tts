"""Orchestrate stages 1–6 for one dialogue without breaking batch runs."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .assemble import assemble_prepared_dialogue
from .backend_info import describe_engine
from .input_contract import validate_dialogue
from .input_records import InputRecord
from .metadata import build_metadata, write_metadata
from .postprocess import apply_telephone_effect
from .qc import QCResult, run_qc
from .render_plan import prepare_dialogue
from .tts_engine import preflight_prepared_dialogue, synthesize_prepared_turns

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    dialogue_id: str | None
    status: str
    error: str | None = None
    qc: QCResult | None = None
    out_dir: Path | None = None
    error_type: str | None = None


async def run_dialogue(
    input_record: InputRecord,
    sidecar: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
    *,
    project_root: Path,
) -> PipelineResult:
    """Render one production InputRecord through the canonical prepared path."""
    dialogue_id = input_record.dialogue_id
    started = time.perf_counter()
    try:
        validate_dialogue(input_record.raw)
        prepared = prepare_dialogue(
            input_record, sidecar, project_root=project_root, config=config
        )
        # This gate deliberately precedes even output-directory creation.
        preflight_prepared_dialogue(prepared, config)

        out_dir = output_root / dialogue_id
        out_dir.mkdir(parents=True, exist_ok=True)
        turn_results = await synthesize_prepared_turns(prepared, out_dir, config)
        full_audio, timings = assemble_prepared_dialogue(prepared, turn_results, config)
        clean_path = out_dir / f"{dialogue_id}_clean.wav"
        full_audio.export(clean_path, format="wav")
        telephone_path = out_dir / f"{dialogue_id}_telephone.wav"
        apply_telephone_effect(full_audio, config).export(telephone_path, format="wav")

        engine_info = describe_engine(config, prepared)
        metadata = build_metadata(
            input_record,
            prepared,
            clean_path,
            telephone_path,
            timings,
            turn_results,
            engine_info,
        )
        write_metadata(metadata, out_dir)
        qc = run_qc(
            input_record,
            prepared,
            turn_results,
            timings,
            clean_path,
            telephone_path,
            metadata,
        )
        logger.info(
            "event=final_dialogue_pipeline_complete dialogue=%s passed=%s "
            "elapsed_sec=%.2f",
            dialogue_id,
            qc.passed,
            time.perf_counter() - started,
        )
        return PipelineResult(
            dialogue_id=dialogue_id,
            status="success" if qc.passed else "failed",
            error=None if qc.passed else f"QC failed: {qc.issues}",
            qc=qc,
            out_dir=out_dir,
            error_type=None if qc.passed else "QCFailure",
        )
    except Exception as error:
        logger.exception(
            "event=final_dialogue_pipeline_failed dialogue=%s elapsed_sec=%.2f",
            dialogue_id,
            time.perf_counter() - started,
        )
        return PipelineResult(
            dialogue_id=dialogue_id,
            status="failed",
            error=f"Final dialogue failed: {error}",
            error_type=type(error).__name__,
        )
