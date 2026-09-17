"""Deterministic, no-inference Controlled TTS v1 render-plan adapter."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .planner import plan_hesitations, plan_within_pauses
from .postprocess import build_postprocess_plan
from .schema import load_contract, normalize_turn


def _warning(code: str, control: str, message: str) -> dict[str, str]:
    return {"code": code, "control": control, "message": message}


def _best_effort(
    normalized: dict[str, Any],
    prefix_tokens: list[str],
    contract: dict[str, Any],
) -> dict[str, Any]:
    requested = normalized["best_effort"]
    fields: dict[str, Any] = {}
    expected = contract["upstream_normalization"]["best_effort_fields"]
    policy = contract["best_effort_policy"]
    joint_rules = policy["joint_rules"]
    statuses = policy["statuses"]
    for field in expected:
        present, value = field in requested, requested.get(field)
        joint = next(
            (
                rule
                for rule in joint_rules
                if rule["field"] == field
                and rule["coarse_affect"] == normalized["coarse_affect"]
                and rule["value"] == value
                and rule["required_token"] in prefix_tokens
            ),
            None,
        )
        if joint is not None:
            fields[field] = {
                "source_present": present,
                "requested": deepcopy(value),
                "status": statuses["jointly_realized"],
                "realization": {
                    "required_token": joint["required_token"],
                    "note": "already jointly represented by required coarse mapping",
                },
            }
        elif value is None or value is False or value == [] or value == "":
            fields[field] = {
                "source_present": present,
                "requested": deepcopy(value),
                "status": statuses["empty_or_absent"],
                "realization": None,
            }
        else:
            fields[field] = {
                "source_present": present,
                "requested": deepcopy(value),
                "status": statuses["unsupported"],
                "realization": None,
            }
    return {"requested": deepcopy(requested), "fields": fields}


def _override(
    normalized: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any] | None:
    for item in contract["combination_overrides"]:
        if all(normalized[key] == value for key, value in item["match"].items()):
            return deepcopy(item)
    return None


def _combination_validation(
    normalized: dict[str, Any],
    override: dict[str, Any] | None,
    contract: dict[str, Any],
) -> tuple[str | None, list[dict[str, str]]]:
    pair = {
        "arousal": normalized["arousal"],
        "coarse_affect": normalized["coarse_affect"],
    }
    if override is not None:
        return override.get("validation_status"), []
    for item in contract["unseen_combination_policy"]["explicit_combination_evidence"]:
        if item.get("match") == pair:
            return item.get("validation_status"), []
    status = contract["unseen_combination_policy"]["validation_status_when_unlisted"]
    return status, [
        _warning(
            status,
            "arousal+coarse_affect",
            (
                "No explicit canonical combination override or combination-level "
                "evidence classification exists."
            ),
        )
    ]


def map_turn_to_higgs(
    turn: Mapping[str, Any],
    *,
    dialogue_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one deterministic v1 render plan; never call Higgs or process audio."""
    contract = load_contract()
    normalized = normalize_turn(turn, dialogue_context=dialogue_context)
    override = _override(normalized, contract)
    arousal_value = contract["controls"]["arousal"]["values"][normalized["arousal"]]
    affect_value = contract["controls"]["coarse_affect"]["values"][
        normalized["coarse_affect"]
    ]
    if override is not None:
        prefix_tokens = list(override["prefix_tokens"])
        arousal_realization = {
            "requested": normalized["arousal"],
            "status": override["status"],
            "token": prefix_tokens[0],
            "joint_realization_group": override["joint_realization_group"],
        }
        affect_realization = {
            "requested": normalized["coarse_affect"],
            "status": override["status"],
            "token": prefix_tokens[0],
            "joint_realization_group": override["joint_realization_group"],
        }
        override_meta: dict[str, Any] | None = {
            "id": override["id"],
            "matched": deepcopy(override["match"]),
            "prefix_tokens": prefix_tokens,
            "status": override["status"],
            "joint_realization_group": deepcopy(override["joint_realization_group"]),
            "replaces": deepcopy(override["replaces"]),
        }
    else:
        prefix_tokens = [
            token
            for token in (arousal_value["token"], affect_value["token"])
            if token is not None
        ]
        arousal_realization = {
            "requested": normalized["arousal"],
            "status": arousal_value["status"],
            "token": arousal_value["token"],
        }
        affect_realization = {
            "requested": normalized["coarse_affect"],
            "status": affect_value["status"],
            "token": affect_value["token"],
        }
        override_meta = None
    hesitation = plan_hesitations(
        normalized["text"], normalized["hesitation_count"], contract
    )
    within = plan_within_pauses(
        hesitation["text"],
        normalized["pause_within_count"],
        hesitation["reserved_spans"],
        contract,
    )
    if hesitation["inserted_count"] != normalized["hesitation_count"]:
        raise AssertionError(
            "hesitation planner did not preserve the requested event count"
        )
    if within["inserted_count"] != normalized["pause_within_count"]:
        raise AssertionError(
            "pause-within planner did not preserve the requested native token count"
        )
    pause_value = contract["controls"]["pause_before"]["values"][
        normalized["pause_before"]
    ]
    rate_value = contract["controls"]["speaking_rate"]["values"][normalized["rate"]]
    validation_status, combination_warnings = _combination_validation(
        normalized, override, contract
    )
    warnings = []
    if normalized["speaker_id"] is None:
        warnings.append(
            _warning(
                "speaker_id_unavailable",
                "logical_speaker",
                (
                    "No dialogue_context was supplied; no speaker reference is "
                    "resolved by this package."
                ),
            )
        )
    if normalized["missing_best_effort"]:
        warnings.append(
            _warning(
                "best_effort_fields_missing",
                "best_effort",
                ("Missing fields were preserved as absent and do not block rendering."),
            )
        )
    if normalized["provenance"].get("known_issue"):
        warnings.append(
            _warning(
                "downstream_exclusion_required",
                "provenance",
                ("Known-issue metadata is preserved; LP/downstream owns filtering."),
            )
        )
    warnings.extend(hesitation["warnings"])
    warnings.extend(within["warnings"])
    warnings.extend(combination_warnings)
    best_effort = _best_effort(normalized, prefix_tokens, contract)
    realization = {
        "speaking_rate": {
            "requested": normalized["rate"],
            "status": rate_value["status"],
            "native_token": None,
            "atempo_factor": rate_value["atempo_factor"],
        },
        "arousal": arousal_realization,
        "coarse_affect": affect_realization,
        "pause_before": {
            "requested": normalized["pause_before"],
            "status": pause_value["status"],
            "milliseconds": pause_value["milliseconds"],
        },
        "pause_within_count": {
            "requested": normalized["pause_within_count"],
            "status": within["status"],
            "inserted_native_pause_tokens": within["inserted_count"],
            "caveat": (
                "planned native tokens are not an exact total audible pause count"
            ),
        },
        "hesitation_count": {
            "requested": normalized["hesitation_count"],
            "status": hesitation["status"],
            "inserted_events": hesitation["inserted_count"],
            "caveat": "generic placement semantics remain provisional",
        },
    }
    native_rate_tokens = [
        value["native_token"]
        for value in contract["controls"]["speaking_rate"]["values"].values()
        if value["native_token"] is not None
    ]
    return {
        "mapping_version": contract["mapping_version"],
        "release_status": contract["release_status"],
        "source": {
            "dialogue_id": normalized["dialogue_id"],
            "turn_id": normalized["turn_id"],
        },
        "logical_speaker": {
            "upstream_role": normalized["upstream_role"],
            "role": normalized["role"],
            "speaker_id": normalized["speaker_id"],
        },
        "requested": {
            "required": normalized["required_raw"],
            "best_effort": deepcopy(normalized["best_effort"]),
        },
        "normalized": {
            key: normalized[key]
            for key in (
                "role",
                "rate",
                "arousal",
                "coarse_affect",
                "pause_before",
                "pause_within_count",
                "hesitation_count",
            )
        },
        "higgs": {
            "prefix_tokens": prefix_tokens,
            "text": within["text"],
            "model_input": "".join(prefix_tokens) + within["text"],
            "native_rate_tokens": native_rate_tokens,
            "native_pause_token_count": within["inserted_count"],
            "synthesis_call_count": contract["model"]["synthesis_call_count"],
        },
        "postprocess": build_postprocess_plan(rate_value, pause_value),
        "realization": realization,
        "combination_override": override_meta,
        "validation_status": validation_status,
        "planner": {
            "hesitation": {
                key: value
                for key, value in hesitation.items()
                if key != "reserved_spans"
            },
            "pause_within": within,
        },
        "best_effort": best_effort,
        "provenance": normalized["provenance"],
        "warnings": warnings,
    }
