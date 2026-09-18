"""Backend-neutral final records and cached backend execution plans."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .controlled_tts import map_turn_to_higgs
from .controlled_tts.planner import plan_hesitations
from .controlled_tts.schema import load_contract, normalize_turn
from .cosyvoice_controls import COSYVOICE_SEMANTIC_RATES, resolve_cosyvoice_controls
from .final_input import SchemaFamily, detect_schema_family, validate_final_dialogue
from .final_references import FinalReferenceError, SelectedReference, selected_reference
from .input_records import InputRecord

_RATE_FACTORS = {"slow": 0.85, "normal": None, "fast": 1.15}
_PAUSE_BEFORE_MS = {"none": 0, "short": 500, "long": 900}


class RenderPlanError(ValueError):
    """Raised when final input and its sidecar cannot form an executable plan."""


@dataclass(frozen=True)
class CanonicalTurn:
    """Final source semantics plus production speaker assignment, without model fields."""

    ordinal: int
    source_turn_id: int | str
    text: str
    labels: Any
    upstream_role: str
    logical_role: str
    upstream_scenario_speaker_id: str
    render_speaker_id: str
    required: dict[str, Any]
    best_effort: dict[str, Any]
    rate: str
    arousal: str
    affect: str
    pause_before: str
    pause_before_ms: int
    pause_within_count: int
    hesitation_count: int


@dataclass(frozen=True)
class CanonicalDialogue:
    dialogue_id: str
    record_sha256: str
    turns: tuple[CanonicalTurn, ...]
    assignment_sha256: str
    registry_sha256: str
    active_speakers_sha256: str


@dataclass(frozen=True, init=False)
class PreparedTurn:
    """Cached frozen Higgs plan; the name is retained for Phase-1 compatibility."""

    ordinal: int
    source_turn_id: int | str
    upstream_role: str
    logical_role: str
    upstream_scenario_speaker_id: str
    render_speaker_id: str
    reference_wav: str
    resolved_reference_wav: Path
    reference_sha256: str
    _plan: dict[str, Any] = field(repr=False)

    def __init__(
        self,
        *,
        ordinal: int,
        source_turn_id: int | str,
        upstream_role: str,
        logical_role: str,
        upstream_scenario_speaker_id: str,
        render_speaker_id: str,
        reference_wav: str,
        resolved_reference_wav: Path,
        reference_sha256: str,
        plan: dict[str, Any],
    ) -> None:
        _validate_higgs_plan(
            plan,
            turn_id=source_turn_id,
            upstream_role=upstream_role,
            logical_role=logical_role,
            scenario_speaker_id=upstream_scenario_speaker_id,
        )
        values = locals()
        for name in self.__dataclass_fields__:
            if name != "_plan":
                object.__setattr__(self, name, values[name])
        object.__setattr__(self, "_plan", deepcopy(plan))

    @property
    def plan(self) -> dict[str, Any]:
        return deepcopy(self._plan)

    @property
    def backend(self) -> str:
        return "higgs"

    @property
    def model_input(self) -> str:
        return self._plan["higgs"]["model_input"]

    @property
    def synthesis_call_count(self) -> int:
        return self._plan["higgs"]["synthesis_call_count"]

    @property
    def native_pause_token_count(self) -> int:
        return self._plan["higgs"]["native_pause_token_count"]

    @property
    def rate(self) -> str:
        return self._plan["normalized"]["rate"]

    @property
    def rate_plan(self) -> dict[str, Any]:
        return deepcopy(self._plan["postprocess"]["atempo"])

    @property
    def pause_before_ms(self) -> int:
        return self._plan["postprocess"]["pause_before_ms"]

    @property
    def approved_reference(self) -> dict[str, Any]:
        return {
            "render_speaker_id": self.render_speaker_id,
            "reference_wav": self.reference_wav,
            "resolved_reference_wav": str(self.resolved_reference_wav),
            "sha256": self.reference_sha256,
        }


HiggsPreparedTurn = PreparedTurn


@dataclass(frozen=True, init=False)
class CosyVoicePreparedTurn:
    """Cached final-schema CosyVoice3 request plan."""

    ordinal: int
    source_turn_id: int | str
    upstream_role: str
    logical_role: str
    upstream_scenario_speaker_id: str
    render_speaker_id: str
    prompt_wav: str
    resolved_prompt_wav: Path
    prompt_text: str
    reference_sha256: str
    _plan: dict[str, Any] = field(repr=False)

    def __init__(
        self,
        *,
        ordinal: int,
        source_turn_id: int | str,
        upstream_role: str,
        logical_role: str,
        upstream_scenario_speaker_id: str,
        render_speaker_id: str,
        prompt_wav: str,
        resolved_prompt_wav: Path,
        prompt_text: str,
        reference_sha256: str,
        plan: dict[str, Any],
    ) -> None:
        _validate_cosyvoice_plan(plan, source_turn_id)
        values = locals()
        for name in self.__dataclass_fields__:
            if name != "_plan":
                object.__setattr__(self, name, values[name])
        object.__setattr__(self, "_plan", deepcopy(plan))

    @property
    def plan(self) -> dict[str, Any]:
        return deepcopy(self._plan)

    @property
    def backend(self) -> str:
        return "cosyvoice"

    @property
    def pause_before_ms(self) -> int:
        return self._plan["postprocess"]["pause_before_ms"]

    @property
    def rate(self) -> str:
        return self._plan["normalized"]["rate"]

    @property
    def approved_reference(self) -> dict[str, Any]:
        return {
            "render_speaker_id": self.render_speaker_id,
            "prompt_wav": self.prompt_wav,
            "resolved_prompt_wav": str(self.resolved_prompt_wav),
            "prompt_text": self.prompt_text,
            "sha256": self.reference_sha256,
        }

    def worker_request(self, output_path: Path) -> dict[str, Any]:
        cosy = self._plan["cosyvoice"]
        request = {
            "text": cosy["text"],
            "prompt_text": cosy["prompt_text"],
            "prompt_wav": str(self.resolved_prompt_wav),
            "output_path": str(output_path),
            "speed": cosy["speed"],
            "mode": cosy["inference_mode"],
        }
        if cosy["instruction"] is not None:
            request["instruction"] = cosy["instruction"]
        return request


@dataclass(frozen=True)
class PreparedDialogue:
    dialogue_id: str
    record_sha256: str
    turns: tuple[PreparedTurn | CosyVoicePreparedTurn, ...]
    assignment_sha256: str
    registry_sha256: str
    active_speakers_sha256: str
    engine: str = "higgs"


@dataclass(frozen=True)
class TurnRenderResult:
    ordinal: int
    source_turn_id: int | str
    output_path: Path
    synthesis_status: str
    rate_status: str


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
    if detect_schema_family(raw) is not SchemaFamily.FINAL_NESTED:
        raise RenderPlanError("InputRecord is not a final nested dialogue")
    validated = validate_final_dialogue(raw)
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


def _validate_higgs_plan(
    plan: dict[str, Any],
    *,
    turn_id: int | str,
    upstream_role: str,
    logical_role: str,
    scenario_speaker_id: str,
) -> None:
    try:
        higgs, normalized = plan["higgs"], plan["normalized"]
        requested, postprocess = plan["requested"]["required"], plan["postprocess"]
        realization, planner = plan["realization"], plan["planner"]
        logical_speaker, source = plan["logical_speaker"], plan["source"]
    except (KeyError, TypeError) as error:
        raise RenderPlanError(
            f"Frozen plan is missing execution field: {error}"
        ) from error
    if plan.get("mapping_version") != "controlled_tts_v1":
        raise RenderPlanError("Frozen plan mapping_version is not controlled_tts_v1")
    if (
        not isinstance(higgs.get("model_input"), str)
        or not higgs["model_input"].strip()
    ):
        raise RenderPlanError("plan.higgs.model_input must be a nonblank string")
    if (
        isinstance(higgs.get("synthesis_call_count"), bool)
        or higgs.get("synthesis_call_count") != 1
    ):
        raise RenderPlanError("plan.higgs.synthesis_call_count must equal 1")
    if higgs.get("native_rate_tokens") != []:
        raise RenderPlanError("plan.higgs.native_rate_tokens must be empty")
    rate = normalized.get("rate")
    expected_factor = _RATE_FACTORS.get(rate, object())
    atempo = _mapping(postprocess.get("atempo"), "plan.postprocess.atempo")
    rate_realization = _mapping(
        realization.get("speaking_rate"), "plan.realization.speaking_rate"
    )
    if rate not in _RATE_FACTORS or not (
        requested.get("rate") == rate
        and postprocess.get("atempo_factor") == expected_factor
        and atempo.get("factor") == expected_factor
        and atempo.get("enabled") is (expected_factor is not None)
        and atempo.get("processor") == "ffmpeg_atempo"
        and rate_realization.get("requested") == rate
        and rate_realization.get("native_token") is None
        and rate_realization.get("atempo_factor") == expected_factor
    ):
        raise RenderPlanError(f"Prepared rate plan is inconsistent for {rate!r}")
    pause = normalized.get("pause_before")
    expected_pause = _PAUSE_BEFORE_MS.get(pause, -1)
    pause_realization = _mapping(
        realization.get("pause_before"), "plan.realization.pause_before"
    )
    if pause not in _PAUSE_BEFORE_MS or not (
        requested.get("pause_before") == pause
        and postprocess.get("pause_before_ms") == expected_pause
        and pause_realization.get("requested") == pause
        and pause_realization.get("milliseconds") == expected_pause
    ):
        raise RenderPlanError(
            f"Prepared pause-before plan is inconsistent for {pause!r}"
        )
    native_count = higgs.get("native_pause_token_count")
    if (
        isinstance(native_count, bool)
        or not isinstance(native_count, int)
        or native_count < 0
    ):
        raise RenderPlanError(
            "plan.higgs.native_pause_token_count must be non-negative"
        )
    if not (
        planner.get("pause_within", {}).get("inserted_count") == native_count
        and realization.get("pause_within_count", {}).get(
            "inserted_native_pause_tokens"
        )
        == native_count
        and normalized.get("pause_within_count") == native_count
    ):
        raise RenderPlanError("Prepared native pause counts are inconsistent")
    if source.get("turn_id") != turn_id:
        raise RenderPlanError("Frozen plan source turn_id does not match source turn")
    if not (
        logical_speaker.get("upstream_role") == upstream_role
        and logical_speaker.get("role") == logical_role
        and logical_speaker.get("speaker_id") == scenario_speaker_id
        and normalized.get("role") == logical_role
    ):
        raise RenderPlanError("Frozen plan speaker semantics do not match source turn")


def _validate_cosyvoice_plan(plan: dict[str, Any], source_turn_id: int | str) -> None:
    try:
        cosy = plan["cosyvoice"]
        normalized = plan["normalized"]
        requested = plan["requested"]["required"]
        realization = plan["realization"]
        capabilities = plan["capabilities"]
    except (KeyError, TypeError) as error:
        raise RenderPlanError(
            f"CosyVoice plan is missing execution field: {error}"
        ) from error
    if (
        plan.get("backend") != "cosyvoice"
        or plan.get("source", {}).get("turn_id") != source_turn_id
    ):
        raise RenderPlanError("CosyVoice plan identity is inconsistent")
    if not isinstance(cosy.get("text"), str) or not cosy["text"].strip():
        raise RenderPlanError("CosyVoice plan text must be nonblank")
    rate = normalized.get("rate")
    expected_speed = COSYVOICE_SEMANTIC_RATES.get(rate)
    speaking_rate = realization.get("speaking_rate", {})
    if (
        expected_speed is None
        or requested.get("rate") != rate
        or cosy.get("speed") != expected_speed
        or speaking_rate.get("requested") != rate
        or speaking_rate.get("speed") != expected_speed
    ):
        raise RenderPlanError("CosyVoice rate and speed plan is inconsistent")
    pause_count = normalized.get("pause_within_count")
    pause_realization = realization.get("pause_within_count", {})
    expected_status = "not_required" if pause_count == 0 else "unsupported"
    if (
        isinstance(pause_count, bool)
        or not isinstance(pause_count, int)
        or pause_count < 0
        or requested.get("pause_within") != pause_count
        or capabilities.get("pause_within") != "fail_closed_when_requested"
        or pause_realization.get("requested") != pause_count
        or pause_realization.get("status") != expected_status
        or pause_realization.get("method") is not None
        or (
            pause_count > 0
            and (
                not isinstance(pause_realization.get("reason"), str)
                or not pause_realization["reason"].strip()
            )
        )
    ):
        raise RenderPlanError("CosyVoice pause_within fail-closed plan is inconsistent")


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
        turns.append(
            PreparedTurn(
                ordinal=turn.ordinal,
                source_turn_id=turn.source_turn_id,
                upstream_role=turn.upstream_role,
                logical_role=turn.logical_role,
                upstream_scenario_speaker_id=turn.upstream_scenario_speaker_id,
                render_speaker_id=turn.render_speaker_id,
                reference_wav=reference.declared_path,
                resolved_reference_wav=reference.resolved_path,
                reference_sha256=reference.sha256,
                plan=map_turn_to_higgs(source, dialogue_context=input_record.raw),
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
                plan=_cosyvoice_plan(turn, reference_snapshot),
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
    config: dict[str, Any],
    *,
    project_root: Path,
) -> PreparedDialogue:
    """Dispatch one canonical final dialogue to exactly one selected backend plan."""
    canonical = canonicalize_dialogue(input_record, sidecar)
    engine = config.get("tts", {}).get("engine")
    if engine == "higgs":
        return _higgs_dialogue(canonical, input_record, sidecar, project_root)
    if engine == "cosyvoice":
        return _cosyvoice_dialogue(canonical, sidecar, project_root)
    raise RenderPlanError(
        f"Final preparation is not implemented for tts.engine {engine!r}"
    )


def prepare_final_dialogue(
    input_record: InputRecord,
    sidecar: dict[str, Any],
    *,
    project_root: Path,
    config: dict[str, Any] | None = None,
) -> PreparedDialogue:
    """Compatibility wrapper; omitted config retains the frozen Higgs API."""
    if config is not None:
        return prepare_dialogue(
            input_record, sidecar, config, project_root=project_root
        )
    canonical = canonicalize_dialogue(input_record, sidecar)
    return _higgs_dialogue(canonical, input_record, sidecar, project_root)
