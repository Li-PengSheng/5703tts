"""Prepare final Controlled TTS v1 records for explicit execution."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .controlled_tts import map_turn_to_higgs
from .final_input import SchemaFamily, detect_schema_family, validate_final_dialogue
from .input_records import InputRecord

_RATE_FACTORS = {"slow": 0.85, "normal": None, "fast": 1.15}
_PAUSE_BEFORE_MS = {"none": 0, "short": 500, "long": 900}


class RenderPlanError(ValueError):
    """Raised when final input and its sidecar cannot form an executable plan."""


@dataclass(frozen=True, init=False)
class PreparedTurn:
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
        _validate_plan(
            plan,
            turn_id=source_turn_id,
            upstream_role=upstream_role,
            logical_role=logical_role,
            scenario_speaker_id=upstream_scenario_speaker_id,
        )
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "source_turn_id", source_turn_id)
        object.__setattr__(self, "upstream_role", upstream_role)
        object.__setattr__(self, "logical_role", logical_role)
        object.__setattr__(
            self, "upstream_scenario_speaker_id", upstream_scenario_speaker_id
        )
        object.__setattr__(self, "render_speaker_id", render_speaker_id)
        object.__setattr__(self, "reference_wav", reference_wav)
        object.__setattr__(self, "resolved_reference_wav", resolved_reference_wav)
        object.__setattr__(self, "reference_sha256", reference_sha256)
        object.__setattr__(self, "_plan", deepcopy(plan))

    @property
    def plan(self) -> dict[str, Any]:
        return deepcopy(self._plan)

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


@dataclass(frozen=True)
class PreparedDialogue:
    dialogue_id: str
    record_sha256: str
    turns: tuple[PreparedTurn, ...]
    assignment_sha256: str
    registry_sha256: str
    active_speakers_sha256: str


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


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


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


def _validate_plan(
    plan: dict[str, Any],
    *,
    turn_id: int | str,
    upstream_role: str,
    logical_role: str,
    scenario_speaker_id: str,
) -> None:
    try:
        higgs = plan["higgs"]
        normalized = plan["normalized"]
        requested = plan["requested"]["required"]
        postprocess = plan["postprocess"]
        realization = plan["realization"]
        planner = plan["planner"]
        logical_speaker = plan["logical_speaker"]
        source = plan["source"]
    except (KeyError, TypeError) as error:
        raise RenderPlanError(
            f"Frozen plan is missing execution field: {error}"
        ) from error

    if plan.get("mapping_version") != "controlled_tts_v1":
        raise RenderPlanError("Frozen plan mapping_version is not controlled_tts_v1")

    model_input = higgs.get("model_input")
    if not isinstance(model_input, str) or not model_input.strip():
        raise RenderPlanError("plan.higgs.model_input must be a nonblank string")
    call_count = higgs.get("synthesis_call_count")
    if isinstance(call_count, bool) or call_count != 1:
        raise RenderPlanError("plan.higgs.synthesis_call_count must equal 1")
    if higgs.get("native_rate_tokens") != []:
        raise RenderPlanError("plan.higgs.native_rate_tokens must be empty")

    rate = normalized.get("rate")
    if rate not in _RATE_FACTORS:
        raise RenderPlanError(f"Unsupported prepared rate: {rate!r}")
    expected_factor = _RATE_FACTORS[rate]
    atempo = _mapping(postprocess.get("atempo"), "plan.postprocess.atempo")
    rate_realization = _mapping(
        realization.get("speaking_rate"), "plan.realization.speaking_rate"
    )
    if (
        requested.get("rate") != rate
        or postprocess.get("atempo_factor") != expected_factor
        or atempo.get("factor") != expected_factor
        or atempo.get("enabled") is (expected_factor is None)
        or atempo.get("processor") != "ffmpeg_atempo"
        or rate_realization.get("requested") != rate
        or rate_realization.get("native_token") is not None
        or rate_realization.get("atempo_factor") != expected_factor
    ):
        raise RenderPlanError(f"Prepared rate plan is inconsistent for {rate!r}")

    pause = normalized.get("pause_before")
    if pause not in _PAUSE_BEFORE_MS:
        raise RenderPlanError(f"Unsupported prepared pause_before: {pause!r}")
    expected_pause = _PAUSE_BEFORE_MS[pause]
    pause_realization = _mapping(
        realization.get("pause_before"), "plan.realization.pause_before"
    )
    if (
        requested.get("pause_before") != pause
        or postprocess.get("pause_before_ms") != expected_pause
        or pause_realization.get("requested") != pause
        or pause_realization.get("milliseconds") != expected_pause
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
    if (
        planner.get("pause_within", {}).get("inserted_count") != native_count
        or realization.get("pause_within_count", {}).get("inserted_native_pause_tokens")
        != native_count
        or normalized.get("pause_within_count") != native_count
    ):
        raise RenderPlanError("Prepared native pause counts are inconsistent")

    if source.get("turn_id") != turn_id:
        raise RenderPlanError("Frozen plan source turn_id does not match source turn")
    if (
        logical_speaker.get("upstream_role") != upstream_role
        or logical_speaker.get("role") != logical_role
        or logical_speaker.get("speaker_id") != scenario_speaker_id
        or normalized.get("role") != logical_role
    ):
        raise RenderPlanError("Frozen plan speaker semantics do not match source turn")


def prepare_final_dialogue(
    input_record: InputRecord,
    sidecar: dict[str, Any],
    *,
    project_root: Path,
) -> PreparedDialogue:
    """Validate and map every final turn exactly once into stable snapshots."""
    raw = input_record.raw
    if detect_schema_family(raw) is not SchemaFamily.FINAL_NESTED:
        raise RenderPlanError("InputRecord is not a final nested dialogue")
    validated = validate_final_dialogue(raw)
    if input_record.dialogue_id != validated.dialogue_id:
        raise RenderPlanError("InputRecord dialogue_id does not match its payload")

    entry = _dialogue_entry(sidecar, validated.dialogue_id)
    source = _mapping(entry.get("source"), "sidecar.source")
    sidecar_record_sha = _sha256(
        source.get("record_sha256"), "sidecar.source.record_sha256"
    )
    if sidecar_record_sha != input_record.record_sha256:
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
    expected_roles = set(validated.speaker_identities)
    if set(roles) != expected_roles:
        missing = sorted(expected_roles - set(roles))
        extra = sorted(set(roles) - expected_roles)
        raise RenderPlanError(
            f"sidecar roles are incomplete or unexpected: missing={missing}, extra={extra}"
        )

    prepared_roles: dict[str, dict[str, Any]] = {}
    for logical_role, identity in validated.speaker_identities.items():
        role = _mapping(roles[logical_role], f"sidecar.roles.{logical_role}")
        upstream_role = _nonblank(
            role.get("upstream_role"), f"sidecar.roles.{logical_role}.upstream_role"
        )
        if upstream_role != identity.upstream_role:
            raise RenderPlanError(
                f"sidecar upstream_role mismatch for logical role {logical_role!r}"
            )
        scenario_id = _nonblank(
            role.get("upstream_scenario_speaker_id"),
            f"sidecar.roles.{logical_role}.upstream_scenario_speaker_id",
        )
        if scenario_id != identity.upstream_scenario_speaker_id:
            raise RenderPlanError(
                f"sidecar scenario speaker mismatch for logical role {logical_role!r}"
            )
        render_id = _nonblank(
            role.get("render_speaker_id"),
            f"sidecar.roles.{logical_role}.render_speaker_id",
        )
        if not render_id.startswith("spk_") or render_id == scenario_id:
            raise RenderPlanError(
                f"sidecar render_speaker_id is not a production speaker: {render_id!r}"
            )
        reference = _mapping(
            role.get("higgs_reference"),
            f"sidecar.roles.{logical_role}.higgs_reference",
        )
        declared_path = _nonblank(
            reference.get("reference_wav"),
            f"sidecar.roles.{logical_role}.higgs_reference.reference_wav",
        )
        reference_sha = _sha256(
            reference.get("sha256"),
            f"sidecar.roles.{logical_role}.higgs_reference.sha256",
        )
        resolved = Path(declared_path)
        if not resolved.is_absolute():
            resolved = project_root / resolved
        resolved = resolved.resolve()
        if not resolved.is_file():
            raise RenderPlanError(
                f"Approved Higgs reference file is missing: {resolved}"
            )
        if _sha256_file(resolved) != reference_sha:
            raise RenderPlanError(
                f"Approved Higgs reference SHA-256 mismatch: {resolved}"
            )
        prepared_roles[logical_role] = {
            "upstream_role": upstream_role,
            "scenario_id": scenario_id,
            "render_id": render_id,
            "reference_wav": declared_path,
            "resolved_reference_wav": resolved,
            "reference_sha256": reference_sha,
        }

    prepared_turns: list[PreparedTurn] = []
    for ordinal, turn in enumerate(raw["turns"], start=1):
        source_turn_id = turn["turn_id"]
        if isinstance(source_turn_id, bool) or not isinstance(
            source_turn_id, (int, str)
        ):
            raise RenderPlanError("Final source turn_id must be an integer or string")
        if isinstance(source_turn_id, str) and not source_turn_id.strip():
            raise RenderPlanError("Final source turn_id must not be blank")
        upstream_role = turn["speaker"]
        logical_role = next(
            (
                identity.logical_role
                for identity in validated.speaker_identities.values()
                if identity.upstream_role == upstream_role
            ),
            None,
        )
        if logical_role is None:
            raise RenderPlanError(f"Unsupported final upstream role: {upstream_role!r}")
        role = prepared_roles[logical_role]
        if role["upstream_role"] != upstream_role:
            raise RenderPlanError(
                f"sidecar upstream role does not match turn {source_turn_id!r}"
            )
        plan = map_turn_to_higgs(turn, dialogue_context=raw)
        prepared_turns.append(
            PreparedTurn(
                ordinal=ordinal,
                source_turn_id=source_turn_id,
                upstream_role=upstream_role,
                logical_role=logical_role,
                upstream_scenario_speaker_id=role["scenario_id"],
                render_speaker_id=role["render_id"],
                reference_wav=role["reference_wav"],
                resolved_reference_wav=role["resolved_reference_wav"],
                reference_sha256=role["reference_sha256"],
                plan=plan,
            )
        )

    return PreparedDialogue(
        dialogue_id=validated.dialogue_id,
        record_sha256=input_record.record_sha256,
        turns=tuple(prepared_turns),
        assignment_sha256=identities["assignment_sha256"],
        registry_sha256=identities["registry_sha256"],
        active_speakers_sha256=identities["active_speakers_sha256"],
    )
