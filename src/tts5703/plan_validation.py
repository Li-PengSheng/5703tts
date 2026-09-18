"""Validation for cached backend execution plans."""

from typing import Any

from .cosyvoice_controls import COSYVOICE_SEMANTIC_RATES

_RATE_FACTORS = {"slow": 0.85, "normal": None, "fast": 1.15}
_PAUSE_BEFORE_MS = {"none": 0, "short": 500, "long": 900}


class RenderPlanError(ValueError):
    """Raised when source input and its sidecar cannot form an executable plan."""


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RenderPlanError(f"{path} must be an object")
    return value


def validate_higgs_plan(
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


def validate_cosyvoice_plan(plan: dict[str, Any], source_turn_id: int | str) -> None:
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
