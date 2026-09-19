"""Canonical dialogue construction and backend-specific planning."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .backends.cosyvoice_controls import resolve_cosyvoice_controls
from .controlled_tts import map_turn_to_higgs
from .controlled_tts.planner import plan_hesitations
from .controlled_tts.schema import load_contract, normalize_turn
from .input_contract import validate_dialogue
from .input_records import InputRecord
from .plan_validation import (
    RenderPlanError,
    validate_cosyvoice_plan,
    validate_higgs_plan,
)
from .render_models import (
    CanonicalDialogue,
    CanonicalTurn,
    CosyVoicePreparedTurn,
    HiggsPreparedTurn,
    PreparedDialogue,
)
from .speaker_references import (
    FinalReferenceError,
    SelectedReference,
    selected_reference,
)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RenderPlanError(f"{path} must be an object")
    return value


def _nonblank(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RenderPlanError(f"{path} must be a nonblank string")
    return value.strip()


def _sha256(value: Any, path: str) -> str:
    value = _nonblank(value, path)
    if len(value) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in value
    ):
        raise RenderPlanError(f"{path} must be exactly 64 hexadecimal characters")
    return value.lower()


def _dialogue_entry(sidecar: dict[str, Any], dialogue_id: str) -> dict[str, Any]:
    entries = sidecar.get("dialogues")
    if entries is None:
        entry = sidecar
    else:
        if not isinstance(entries, list):
            raise RenderPlanError("sidecar.dialogues must be a list")
        matches = [
            item
            for item in entries
            if isinstance(item, dict) and item.get("dialogue_id") == dialogue_id
        ]
        if len(matches) != 1:
            raise RenderPlanError(
                f"sidecar must contain exactly one entry for dialogue_id {dialogue_id!r}"
            )
        entry = matches[0]
    entry_id = _nonblank(entry.get("dialogue_id"), "sidecar.dialogue_id")
    if entry_id != dialogue_id:
        raise RenderPlanError(
            f"sidecar dialogue_id {entry_id!r} does not match {dialogue_id!r}"
        )
    return entry


def _source_turn_id(turn: dict[str, Any]) -> int | str:
    value = turn["turn_id"]
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise RenderPlanError("Final source turn_id must be an integer or string")
    if isinstance(value, str) and not value.strip():
        raise RenderPlanError("Final source turn_id must not be blank")
    return value


def canonicalize_dialogue(
    input_record: InputRecord, sidecar: dict[str, Any]
) -> CanonicalDialogue:
    """Normalize final source semantics once, without introducing backend fields."""
    raw = input_record.raw
    validated = validate_dialogue(raw)
    if input_record.dialogue_id != validated.dialogue_id:
        raise RenderPlanError("InputRecord dialogue_id does not match its payload")
    entry = _dialogue_entry(sidecar, validated.dialogue_id)
    source = _mapping(entry.get("source"), "sidecar.source")
    if (
        _sha256(source.get("record_sha256"), "sidecar.source.record_sha256")
        != input_record.record_sha256
    ):
        raise RenderPlanError("sidecar source record_sha256 does not match InputRecord")
    identities = {
        key: _sha256(sidecar.get(key), f"sidecar.{key}")
        for key in (
            "assignment_sha256",
            "registry_sha256",
            "active_speakers_sha256",
        )
    }
    roles = _mapping(entry.get("roles"), "sidecar.roles")
    if set(roles) != set(validated.speaker_identities):
        raise RenderPlanError(
            "sidecar roles are incomplete or unexpected: "
            f"missing={sorted(set(validated.speaker_identities) - set(roles))}, "
            f"extra={sorted(set(roles) - set(validated.speaker_identities))}"
        )

    role_rows: dict[str, dict[str, Any]] = {}
    for logical_role, identity in validated.speaker_identities.items():
        role = _mapping(roles[logical_role], f"sidecar.roles.{logical_role}")
        upstream_role = _nonblank(
            role.get("upstream_role"),
            f"sidecar.roles.{logical_role}.upstream_role",
        )
        scenario_id = _nonblank(
            role.get("upstream_scenario_speaker_id"),
            f"sidecar.roles.{logical_role}.upstream_scenario_speaker_id",
        )
        render_id = _nonblank(
            role.get("render_speaker_id"),
            f"sidecar.roles.{logical_role}.render_speaker_id",
        )
        if upstream_role != identity.upstream_role:
            raise RenderPlanError(
                f"sidecar upstream_role mismatch for logical role {logical_role!r}"
            )
        if scenario_id != identity.upstream_scenario_speaker_id:
            raise RenderPlanError(
                f"sidecar scenario speaker mismatch for logical role {logical_role!r}"
            )
        if not render_id.startswith("spk_") or render_id == scenario_id:
            raise RenderPlanError(
                f"sidecar render_speaker_id is not a production speaker: {render_id!r}"
            )
        role_rows[logical_role] = {**role, "render_speaker_id": render_id}

    contract = load_contract()
    pause_values = contract["controls"]["pause_before"]["values"]
    canonical_turns = []
    for ordinal, source_turn in enumerate(raw["turns"], start=1):
        normalized = normalize_turn(source_turn, dialogue_context=raw)
        logical_role = normalized["role"]
        role = role_rows[logical_role]
        canonical_turns.append(
            CanonicalTurn(
                ordinal=ordinal,
                source_turn_id=_source_turn_id(source_turn),
                text=normalized["text"],
                labels=deepcopy(source_turn["labels"]),
                upstream_role=normalized["upstream_role"],
                logical_role=logical_role,
                upstream_scenario_speaker_id=normalized["speaker_id"],
                render_speaker_id=role["render_speaker_id"],
                required=deepcopy(normalized["required_raw"]),
                best_effort=deepcopy(normalized["best_effort"]),
                rate=normalized["rate"],
                arousal=normalized["arousal"],
                affect=normalized["coarse_affect"],
                pause_before=normalized["pause_before"],
                pause_before_ms=pause_values[normalized["pause_before"]][
                    "milliseconds"
                ],
                pause_within_count=normalized["pause_within_count"],
                hesitation_count=normalized["hesitation_count"],
            )
        )
    return CanonicalDialogue(
        dialogue_id=validated.dialogue_id,
        record_sha256=input_record.record_sha256,
        turns=tuple(canonical_turns),
        assignment_sha256=identities["assignment_sha256"],
        registry_sha256=identities["registry_sha256"],
        active_speakers_sha256=identities["active_speakers_sha256"],
    )


def _resolved_reference(
    role: dict[str, Any],
    engine: str,
    logical_role: str,
    project_root: Path,
) -> SelectedReference:
    try:
        return selected_reference(
            role,
            engine,
            logical_role,
            project_root=project_root,
            verify_file=True,
        )
    except FinalReferenceError as error:
        raise RenderPlanError(str(error)) from error


def _sidecar_roles(sidecar: dict[str, Any], dialogue_id: str) -> dict[str, Any]:
    return _mapping(_dialogue_entry(sidecar, dialogue_id).get("roles"), "sidecar.roles")


def _higgs_dialogue(
    canonical: CanonicalDialogue,
    input_record: InputRecord,
    sidecar: dict[str, Any],
    project_root: Path,
) -> PreparedDialogue:
    roles = _sidecar_roles(sidecar, canonical.dialogue_id)
    turns = []
    for source, turn in zip(input_record.raw["turns"], canonical.turns, strict=True):
        reference = _resolved_reference(
            roles[turn.logical_role],
            "higgs",
            turn.logical_role,
            project_root,
        )
        assert reference.resolved_path is not None
        plan = map_turn_to_higgs(source, dialogue_context=input_record.raw)
        validate_higgs_plan(
            plan,
            turn_id=turn.source_turn_id,
            upstream_role=turn.upstream_role,
            logical_role=turn.logical_role,
            scenario_speaker_id=turn.upstream_scenario_speaker_id,
        )
        turns.append(
            HiggsPreparedTurn(
                ordinal=turn.ordinal,
                source_turn_id=turn.source_turn_id,
                upstream_role=turn.upstream_role,
                logical_role=turn.logical_role,
                upstream_scenario_speaker_id=turn.upstream_scenario_speaker_id,
                render_speaker_id=turn.render_speaker_id,
                reference_wav=reference.declared_path,
                resolved_reference_wav=reference.resolved_path,
                reference_sha256=reference.sha256,
                plan=plan,
            )
        )
    return PreparedDialogue(
        canonical.dialogue_id,
        canonical.record_sha256,
        tuple(turns),
        canonical.assignment_sha256,
        canonical.registry_sha256,
        canonical.active_speakers_sha256,
        "higgs",
    )


def _cosyvoice_plan(turn: CanonicalTurn, reference: dict[str, Any]) -> dict[str, Any]:
    contract = load_contract()
    hesitation = plan_hesitations(turn.text, turn.hesitation_count, contract)
    controls = resolve_cosyvoice_controls(turn.rate, turn.arousal, turn.affect)
    pause_status = "not_required" if turn.pause_within_count == 0 else "unsupported"
    best_effort = {
        field: {
            "requested": deepcopy(turn.best_effort.get(field)),
            "status": (
                "not_realized"
                if turn.best_effort.get(field) in (None, False, "")
                or turn.best_effort.get(field) == []
                else "unsupported"
            ),
            "realization": None,
        }
        for field in contract["upstream_normalization"]["best_effort_fields"]
    }
    return {
        "backend": "cosyvoice",
        "mapping": deepcopy(controls["mapping"]),
        "source": {"turn_id": turn.source_turn_id},
        "logical_speaker": {
            "upstream_role": turn.upstream_role,
            "role": turn.logical_role,
            "speaker_id": turn.upstream_scenario_speaker_id,
            "render_speaker_id": turn.render_speaker_id,
        },
        "requested": {
            "required": deepcopy(turn.required),
            "best_effort": deepcopy(turn.best_effort),
        },
        "normalized": {
            "role": turn.logical_role,
            "rate": turn.rate,
            "arousal": turn.arousal,
            "coarse_affect": turn.affect,
            "pause_before": turn.pause_before,
            "pause_within_count": turn.pause_within_count,
            "hesitation_count": turn.hesitation_count,
        },
        "cosyvoice": {
            "text": hesitation["text"],
            "prompt_wav": reference["prompt_wav"],
            "prompt_text": reference["prompt_text"],
            "reference_sha256": reference["sha256"],
            "speed": controls["speaking_rate"]["speed"],
            "inference_mode": controls["inference_mode"],
            "instruction": controls["resolved_instruction"],
        },
        "postprocess": {
            "pause_before_ms": turn.pause_before_ms,
            "pause_after_ms": 0,
        },
        "realization": {
            "speaking_rate": controls["speaking_rate"],
            "arousal": controls["arousal"],
            "coarse_affect": controls["coarse_affect"],
            "pause_before": {
                "requested": turn.pause_before,
                "status": "pipeline_timing",
                "milliseconds": turn.pause_before_ms,
            },
            "pause_within_count": {
                "requested": turn.pause_within_count,
                "status": pause_status,
                "method": None,
                "reason": (
                    None
                    if turn.pause_within_count == 0
                    else "No evidence-backed deterministic CosyVoice3 realization is available"
                ),
            },
            "hesitation_count": {
                "requested": turn.hesitation_count,
                "status": hesitation["status"],
                "inserted_events": hesitation["inserted_count"],
                "method": "shared_lexical_planner",
            },
        },
        "capabilities": {
            "pause_before": "pipeline_timing",
            "pause_within": "fail_closed_when_requested",
            "hesitations": "shared_lexical_planner",
        },
        "best_effort": {
            "requested": deepcopy(turn.best_effort),
            "fields": best_effort,
        },
        "planner": {
            "hesitation": {
                key: value
                for key, value in hesitation.items()
                if key != "reserved_spans"
            }
        },
    }


def _cosyvoice_dialogue(
    canonical: CanonicalDialogue,
    sidecar: dict[str, Any],
    project_root: Path,
) -> PreparedDialogue:
    roles = _sidecar_roles(sidecar, canonical.dialogue_id)
    turns = []
    for turn in canonical.turns:
        reference = _resolved_reference(
            roles[turn.logical_role],
            "cosyvoice",
            turn.logical_role,
            project_root,
        )
        assert reference.resolved_path is not None
        assert reference.prompt_text is not None
        reference_snapshot = {
            "prompt_wav": reference.declared_path,
            "prompt_text": reference.prompt_text,
            "sha256": reference.sha256,
        }
        plan = _cosyvoice_plan(turn, reference_snapshot)
        validate_cosyvoice_plan(plan, turn.source_turn_id)
        turns.append(
            CosyVoicePreparedTurn(
                ordinal=turn.ordinal,
                source_turn_id=turn.source_turn_id,
                upstream_role=turn.upstream_role,
                logical_role=turn.logical_role,
                upstream_scenario_speaker_id=turn.upstream_scenario_speaker_id,
                render_speaker_id=turn.render_speaker_id,
                prompt_wav=reference.declared_path,
                resolved_prompt_wav=reference.resolved_path,
                prompt_text=reference.prompt_text,
                reference_sha256=reference.sha256,
                plan=plan,
            )
        )
    return PreparedDialogue(
        canonical.dialogue_id,
        canonical.record_sha256,
        tuple(turns),
        canonical.assignment_sha256,
        canonical.registry_sha256,
        canonical.active_speakers_sha256,
        "cosyvoice",
    )


def prepare_dialogue(
    input_record: InputRecord,
    sidecar: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    project_root: Path,
) -> PreparedDialogue:
    """Dispatch one canonical dialogue to exactly one selected backend plan."""
    canonical = canonicalize_dialogue(input_record, sidecar)
    engine = "higgs" if config is None else config.get("tts", {}).get("engine")
    if engine == "higgs":
        return _higgs_dialogue(canonical, input_record, sidecar, project_root)
    if engine == "cosyvoice":
        return _cosyvoice_dialogue(canonical, sidecar, project_root)
    raise RenderPlanError(
        f"Final preparation is not implemented for tts.engine {engine!r}"
    )
