"""Orchestrate stages 1–6 for one dialogue without breaking batch runs."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .assemble import assemble_dialogue
from .metadata import build_metadata, write_metadata
from .postprocess import apply_telephone_effect
from .qc import QCResult, run_qc
from .tts_engine import (
    BackendControlError,
    describe_engine,
    preflight_dialogue_controls,
    synthesize_all_turns,
)
from .validate import ValidationError, load_and_validate

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    dialogue_id: str
    status: str
    error: str | None = None
    qc: QCResult | None = None
    out_dir: Path | None = None


async def run_dialogue(
    json_path: Path, config: dict[str, Any], output_root: Path
) -> PipelineResult:
    """Run one dialogue through validation, synthesis, assembly, export, and QC.

    Expected input/preflight failures are returned without creating an output
    directory. Failures after that point leave completed per-turn or dialogue
    files in place for diagnosis; the current pipeline has no transactional
    cleanup, retry, or resume protocol.
    """
    dialogue_id = json_path.stem
    started = time.perf_counter()
    try:
        logger.info("event=stage_start dialogue=%s stage=validate", dialogue_id)
        dialogue = load_and_validate(json_path, config)
        dialogue_id = dialogue.dialogue_id
        logger.info(
            "event=stage_complete dialogue=%s stage=validate turns=%d",
            dialogue_id,
            len(dialogue.turns),
        )
        # Backend mappings are checked before any audio is rendered; canonical
        # schema validation stays backend-independent.
        preflight_dialogue_controls(dialogue.turns, config)
        out_dir = output_root / dialogue_id
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "event=stage_start dialogue=%s stage=tts engine=%s",
            dialogue_id,
            config["tts"].get("engine"),
        )
        turn_paths = await synthesize_all_turns(dialogue.turns, out_dir, config)
        logger.info(
            "event=stage_complete dialogue=%s stage=tts turn_audio=%s",
            dialogue_id,
            [path.name for path in turn_paths.values()],
        )
        logger.info("event=stage_start dialogue=%s stage=assemble", dialogue_id)
        full_audio, timings = assemble_dialogue(dialogue.turns, turn_paths, config)
        logger.info(
            "event=stage_complete dialogue=%s stage=assemble duration_sec=%.3f",
            dialogue_id,
            len(full_audio) / 1000,
        )
        clean_path = out_dir / f"{dialogue_id}_clean.wav"
        logger.info("event=stage_start dialogue=%s stage=postprocess", dialogue_id)
        full_audio.export(clean_path, format="wav")
        telephone_path = out_dir / f"{dialogue_id}_telephone.wav"
        apply_telephone_effect(full_audio, config).export(telephone_path, format="wav")
        logger.info(
            "event=stage_complete dialogue=%s stage=postprocess clean=%s telephone=%s",
            dialogue_id,
            clean_path.name,
            telephone_path.name,
        )
        logger.info("event=stage_start dialogue=%s stage=metadata", dialogue_id)
        metadata = build_metadata(
            dialogue_id,
            clean_path,
            telephone_path,
            timings,
            turn_paths,
            describe_engine(config),
        )
        metadata_path = write_metadata(metadata, out_dir)
        logger.info(
            "event=stage_complete dialogue=%s stage=metadata path=%s",
            dialogue_id,
            metadata_path.name,
        )
        logger.info("event=stage_start dialogue=%s stage=qc", dialogue_id)
        qc = run_qc(dialogue, out_dir, clean_path, telephone_path, metadata)
        logger.info(
            "event=stage_complete dialogue=%s stage=qc passed=%s checks=%s issues=%s",
            dialogue_id,
            qc.passed,
            qc.checks,
            qc.issues,
        )
        logger.info(
            "event=dialogue_pipeline_complete dialogue=%s elapsed_sec=%.2f",
            dialogue_id,
            time.perf_counter() - started,
        )
        return PipelineResult(
            dialogue_id,
            "success" if qc.passed else "failed",
            None if qc.passed else f"QC failed: {qc.issues}",
            qc,
            out_dir,
        )
    except ValidationError as error:
        logger.warning(
            "event=dialogue_pipeline_failed dialogue=%s stage=validate error=%s",
            dialogue_id,
            error,
        )
        return PipelineResult(dialogue_id, "failed", f"Input validation failed: {error}")
    except BackendControlError as error:
        # Expected compatibility failure, not a defect: the requested control is
        # schema-valid but the selected backend has no mapping for it, so the
        # actionable message matters more than a traceback.
        logger.warning(
            "event=dialogue_pipeline_failed dialogue=%s stage=backend_preflight "
            "error=%s",
            dialogue_id,
            error,
        )
        return PipelineResult(
            dialogue_id, "failed", f"Backend control preflight failed: {error}"
        )
    except Exception as error:
        logger.exception(
            "event=dialogue_pipeline_failed dialogue=%s elapsed_sec=%.2f",
            dialogue_id,
            time.perf_counter() - started,
        )
        return PipelineResult(dialogue_id, "failed", f"Unexpected error: {error}")
