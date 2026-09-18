"""Backend-neutral canonical and prepared rendering data models."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .plan_validation import validate_cosyvoice_plan, validate_higgs_plan


@dataclass(frozen=True)
class CanonicalTurn:
    """Source semantics plus production speaker assignment, without model fields."""

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
class HiggsPreparedTurn:
    """Cached frozen Higgs execution plan."""

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
        validate_higgs_plan(
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


@dataclass(frozen=True, init=False)
class CosyVoicePreparedTurn:
    """Cached CosyVoice3 execution plan."""

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
        validate_cosyvoice_plan(plan, source_turn_id)
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
    turns: tuple[HiggsPreparedTurn | CosyVoicePreparedTurn, ...]
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
