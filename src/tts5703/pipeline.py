"""Single-dialogue production orchestration and failure containment.

The batch owns record selection, resume, and stale cleanup. This module owns one
attempt from validated ``InputRecord`` through prepared execution, clean and
telephone audio, provenance metadata, structural QC, and Higgs quality evidence. Exceptions become a
``PipelineResult`` failure so one dialogue does not crash every other record.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio_quality import assess_dialogue_turns, write_quality_sidecar_atomic
from .backends.info import describe_engine
from .input.contract import validate_dialogue
from .input.records import InputRecord
from .render.assemble import assemble_prepared_dialogue
from .render.metadata import build_metadata, write_metadata
from .render.postprocess import apply_telephone_effect
from .render.qc import QCResult, run_qc
from .render_plan import prepare_dialogue
from .tts_engine import preflight_prepared_dialogue, synthesize_prepared_turns

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Batch-visible success or contained failure for one dialogue attempt."""

    dialogue_id: str | None
    status: str
    error: str | None = None
    qc: QCResult | None = None
    out_dir: Path | None = None
    error_type: str | None = None
    quality: dict[str, Any] | None = None


async def run_dialogue(
    input_record: InputRecord,
    sidecar: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
    *,
    project_root: Path,
    render_fingerprint: str | None,
) -> PipelineResult:
    """Run the ordered production path for exactly one dialogue.

    Order is part of the boundary: validate; prepare; preflight every turn;
    create the output directory; synthesize speech-only turn WAVs; assemble and
    write clean audio; derive telephone audio; describe the backend; build and
    write metadata; run structural QC; assess Higgs audio; return
    ``PipelineResult``.

    Output-directory creation follows preflight so an invalid late turn or
    reference cannot leave a newly-created dialogue directory before any
    synthesis request. Exception conversion below preserves batch progress; it
    does not alter the underlying error or QC semantics.
    """
    dialogue_id = input_record.dialogue_id
    started = time.perf_counter()
    try:
        is_higgs = config.get("tts", {}).get("engine") == "higgs"
        if is_higgs and (
            not isinstance(render_fingerprint, str) or not render_fingerprint.strip()
        ):
            return PipelineResult(
                dialogue_id=dialogue_id,
                status="failed",
                error="Higgs render fingerprint is unavailable; synthesis was not started",
                error_type="RenderIdentityUnavailable",
            )
        validate_dialogue(input_record.raw)
        prepared = prepare_dialogue(
            input_record, sidecar, project_root=project_root, config=config
        )
        # Validate the complete execution plan before creating artifacts or
        # spending GPU work on a prefix of an invalid dialogue.
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
        quality = None
        if qc.passed and is_higgs:
            quality = assess_dialogue_turns(
                out_dir,
                dialogue_id,
                render_fingerprint,
                metadata["turns"],
                assessment="new_render",
            )
            write_quality_sidecar_atomic(
                out_dir,
                dialogue_id,
                quality,
                [turn["execution"]["turn_audio"] for turn in metadata["turns"]],
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
            status="success"
            if qc.passed and (quality is None or quality["outcome"] == "pass")
            else "failed",
            error=(
                f"QC failed: {qc.issues}"
                if not qc.passed
                else f"Higgs audio quality {quality['outcome']}: {quality['issues']}"
                if quality and quality["outcome"] != "pass"
                else None
            ),
            qc=qc,
            out_dir=out_dir,
            error_type=(
                "QCFailure"
                if not qc.passed
                else "QualityRejected"
                if quality and quality["outcome"] != "pass"
                else None
            ),
            quality=quality,
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
