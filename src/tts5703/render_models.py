"""Ownership boundary between source semantics and executable backend plans.

Canonical objects own source/render semantics and production speaker assignment,
but no backend-native execution fields. Prepared objects own the exact cached
backend execution plan. Construction validates that plan and deep-copies it;
accessors return defensive copies so downstream code cannot silently diverge
from the plan recorded in metadata, checked by QC, and represented in identity.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .plan_validation import validate_cosyvoice_plan, validate_higgs_plan


@dataclass(frozen=True)
class CanonicalTurn:
    """One source turn normalized for rendering, without backend-native fields.

    ``upstream_role`` (User/Listener), ``logical_role``
    (caller/counsellor), upstream scenario speaker ID, and production
    ``render_speaker_id`` are intentionally separate identities.
    """

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
    """Backend-neutral dialogue plus source and sidecar provenance hashes."""

    dialogue_id: str
    record_sha256: str
    turns: tuple[CanonicalTurn, ...]
    assignment_sha256: str
    registry_sha256: str
    active_speakers_sha256: str


@dataclass(frozen=True, init=False)
class HiggsPreparedTurn:
    """Validated, cached Higgs execution plan for one canonical turn.

    ``__init__`` is an invariant boundary: ``validate_higgs_plan()`` must pass
    before downstream code may trust model_input, one-call synthesis, native
    pause counts, assembly timing, rate_plan, and source/speaker identity.
    """

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
        """Return a copy; callers must not mutate the frozen execution plan."""
        return deepcopy(self._plan)

    @property
    def backend(self) -> str:
        return "higgs"

    @property
    def model_input(self) -> str:
        """Exact cached text/tokens sent to Higgs; execution never remaps it."""
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
        """Frozen FFmpeg atempo plan applied after Higgs synthesis."""
        return deepcopy(self._plan["postprocess"]["atempo"])

    @property
    def pause_before_ms(self) -> int:
        """Assembly-owned leading silence; it is not part of the turn WAV."""
        return self._plan["postprocess"]["pause_before_ms"]

    @property
    def approved_reference(self) -> dict[str, Any]:
        """Selected approved Higgs reference snapshot for metadata/QC."""
        return {
            "render_speaker_id": self.render_speaker_id,
            "reference_wav": self.reference_wav,
            "resolved_reference_wav": str(self.resolved_reference_wav),
            "sha256": self.reference_sha256,
        }


@dataclass(frozen=True, init=False)
class CosyVoicePreparedTurn:
    """Validated, cached CosyVoice3 execution plan for one canonical turn.

    ``__init__`` runs ``validate_cosyvoice_plan()`` before publishing the
    object.  The plan fixes prompt identity, lexical hesitation text, numeric
    speed, inference mode, provisional instruction, and assembly timing.
    """

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
        """Return a defensive copy of the frozen backend execution plan."""
        return deepcopy(self._plan)

    @property
    def backend(self) -> str:
        return "cosyvoice"

    @property
    def pause_before_ms(self) -> int:
        """Assembly-owned leading silence; it is not synthesized by CosyVoice."""
        return self._plan["postprocess"]["pause_before_ms"]

    @property
    def rate(self) -> str:
        return self._plan["normalized"]["rate"]

    @property
    def approved_reference(self) -> dict[str, Any]:
        """Selected CosyVoice prompt WAV/text snapshot for metadata and QC."""
        return {
            "render_speaker_id": self.render_speaker_id,
            "prompt_wav": self.prompt_wav,
            "resolved_prompt_wav": str(self.resolved_prompt_wav),
            "prompt_text": self.prompt_text,
            "sha256": self.reference_sha256,
        }

    def worker_request(self, output_path: Path) -> dict[str, Any]:
        """Materialize the worker request solely from the validated cached plan."""
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
    """One backend's complete immutable-ish execution plan for a dialogue."""

    dialogue_id: str
    record_sha256: str
    turns: tuple[HiggsPreparedTurn | CosyVoicePreparedTurn, ...]
    assignment_sha256: str
    registry_sha256: str
    active_speakers_sha256: str
    engine: str = "higgs"


@dataclass(frozen=True)
class TurnRenderResult:
    """Execution outcome linking a prepared turn to its speech-only WAV."""

    ordinal: int
    source_turn_id: int | str
    output_path: Path
    synthesis_status: str
    rate_status: str
