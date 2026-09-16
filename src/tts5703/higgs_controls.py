"""Pure, CPU-only Controlled TTS v1 mappings for future Higgs synthesis."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
from typing import Any

HIGGS_CONTROL_SOURCE_REPOSITORY = "ginkgoyin/comp5703-tts-experiments"
HIGGS_CONTROL_SOURCE_COMMIT = "7b005135e148fad8a142f39dc8be3144c3e5adcc"
HIGGS_CONTROL_CONTRACT_SHA256 = (
    "4e787a5363d01e5454a8031c59bb1d4bb454ee37d44ccf262d471c7ce30ba799"
)

_CONTRACT_FILE = "controlled_tts_v1.json"
_UNAVAILABLE = "not_represented_by_current_canonical"
_NOT_EXECUTED = "not_executed"


class HiggsControlError(ValueError):
    """Raised when production controls cannot be mapped by Controlled TTS v1."""


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key in Controlled TTS v1 JSON: {key!r}")
        result[key] = value
    return result


@lru_cache(maxsize=1)
def _contract() -> dict[str, Any]:
    data = files("tts5703").joinpath(_CONTRACT_FILE).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != HIGGS_CONTROL_CONTRACT_SHA256:
        raise ValueError(
            "Controlled TTS v1 contract hash mismatch: "
            f"expected {HIGGS_CONTROL_CONTRACT_SHA256}, got {digest}"
        )
    contract = json.loads(data, object_pairs_hook=_no_duplicate_object)
    if (
        contract.get("mapping_version") != "controlled_tts_v1"
        or contract.get("release_status")
        != "PROVISIONAL_PRODUCTION_MAPPING_WEEK6_FREEZE"
    ):
        raise ValueError("Invalid Controlled TTS v1 contract identity")
    return contract


def load_higgs_control_contract() -> dict[str, Any]:
    """Return an isolated copy of the installed frozen mapping contract."""
    return deepcopy(_contract())


def _domain(field: str, value: str, allowed: Any) -> str:
    if value not in allowed:
        raise HiggsControlError(
            f"Unsupported Higgs {field} mapping: {value!r}. "
            f"Currently supported mappings: {', '.join(allowed)}."
        )
    return value


def _optional_domain(field: str, value: str | None, allowed: Any) -> str | None:
    return None if value is None else _domain(field, value, allowed)


def _nonnegative_ms(field: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HiggsControlError(
            f"Higgs {field} must be a non-negative integer; got: {value!r}"
        )
    return value


def _nonblank(field: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HiggsControlError(f"Higgs {field} must be a non-blank string")
    return value


def _combination_override(
    arousal: str | None, coarse_affect: str | None, contract: dict[str, Any]
) -> dict[str, Any] | None:
    if arousal is None or coarse_affect is None:
        return None
    values = {"arousal": arousal, "coarse_affect": coarse_affect}
    for item in contract["combination_overrides"]:
        if all(values[key] == value for key, value in item["match"].items()):
            return deepcopy(item)
    return None


def _combination_validation(
    arousal: str | None,
    coarse_affect: str | None,
    override: dict[str, Any] | None,
    contract: dict[str, Any],
) -> tuple[str | None, list[dict[str, str]]]:
    if arousal is None or coarse_affect is None:
        return None, []
    if override is not None:
        return override.get("validation_status"), []
    pair = {"arousal": arousal, "coarse_affect": coarse_affect}
    policy = contract["unseen_combination_policy"]
    for item in policy["explicit_combination_evidence"]:
        if item.get("match") == pair:
            return item.get("validation_status"), []
    status = policy["validation_status_when_unlisted"]
    return status, [
        {
            "code": status,
            "control": "arousal+coarse_affect",
            "message": (
                "No explicit canonical combination override or combination-level "
                "evidence classification exists."
            ),
        }
    ]


def resolve_higgs_controls(
    text: str,
    rate: str,
    arousal: str | None,
    coarse_affect: str | None,
    pause_before_ms: int,
    pause_after_ms: int,
    speaker_id: str,
) -> dict[str, Any]:
    """Resolve current production semantics without running Higgs or postprocessing."""
    contract = _contract()
    controls = contract["controls"]
    text = _nonblank("text", text)
    speaker_id = _nonblank("speaker_id", speaker_id)
    rate = _domain("rate", rate, controls["speaking_rate"]["domains"])
    arousal = _optional_domain("arousal", arousal, controls["arousal"]["values"])
    coarse_affect = _optional_domain(
        "coarse_affect", coarse_affect, controls["coarse_affect"]["domains"]
    )
    pause_before_ms = _nonnegative_ms("pause_before_ms", pause_before_ms)
    pause_after_ms = _nonnegative_ms("pause_after_ms", pause_after_ms)

    rate_value = controls["speaking_rate"]["values"][rate]
    arousal_value = (
        controls["arousal"]["values"][arousal] if arousal is not None else None
    )
    affect_value = (
        controls["coarse_affect"]["values"][coarse_affect]
        if coarse_affect is not None
        else None
    )
    override = _combination_override(arousal, coarse_affect, contract)
    if override is None:
        prefix_tokens = [
            token
            for token in (
                arousal_value["token"] if arousal_value is not None else None,
                affect_value["token"] if affect_value is not None else None,
            )
            if token is not None
        ]
        arousal_realization = (
            {
                "requested": arousal,
                "status": arousal_value["status"],
                "token": arousal_value["token"],
            }
            if arousal_value is not None
            else {
                "requested": None,
                "method": "not_requested",
                "status": None,
                "token": None,
            }
        )
        affect_realization = (
            {
                "requested": coarse_affect,
                "status": affect_value["status"],
                "token": affect_value["token"],
            }
            if affect_value is not None
            else {
                "requested": None,
                "method": "not_requested",
                "status": None,
                "token": None,
            }
        )
        override_meta = None
    else:
        prefix_tokens = list(override["prefix_tokens"])
        joint = {
            "status": override["status"],
            "token": prefix_tokens[0],
            "joint_realization_group": list(override["joint_realization_group"]),
        }
        arousal_realization = {"requested": arousal, **joint}
        affect_realization = {"requested": coarse_affect, **joint}
        override_meta = {
            "id": override["id"],
            "matched": deepcopy(override["match"]),
            "prefix_tokens": prefix_tokens,
            "status": override["status"],
            "joint_realization_group": list(override["joint_realization_group"]),
            "replaces": list(override["replaces"]),
        }

    validation_status, warnings = _combination_validation(
        arousal, coarse_affect, override, contract
    )
    atempo_factor = rate_value["atempo_factor"]
    unavailable = {
        field: {"availability": _UNAVAILABLE, "realization": None}
        for field in ("pause_within_count", "hesitation_count", "best_effort")
    }
    requested = {
        "rate": rate,
        "arousal": arousal,
        "coarse_affect": coarse_affect,
        "pause_before_ms": pause_before_ms,
        "pause_after_ms": pause_after_ms,
    }
    return {
        "backend": "higgs",
        "mapping_version": contract["mapping_version"],
        "release_status": contract["release_status"],
        "speaker_id": speaker_id,
        "requested": requested,
        "normalized": dict(requested),
        "higgs": {
            "prefix_tokens": prefix_tokens,
            "text": text,
            "model_input": "".join(prefix_tokens) + text,
            "native_rate_tokens": [],
            "synthesis_call_count": contract["model"]["synthesis_call_count"],
        },
        "postprocess": {
            "rate": {
                "processor": "ffmpeg_atempo",
                "enabled": atempo_factor is not None,
                "factor": atempo_factor,
                "execution_status": _NOT_EXECUTED,
            },
            "turn_boundary_pauses": {
                "owner": "pipeline_assembly",
                "pause_before_ms": pause_before_ms,
                "pause_after_ms": pause_after_ms,
                "audio_insertion_planned": False,
            },
        },
        "realization": {
            "speaking_rate": {
                "requested": rate,
                "status": rate_value["status"],
                "native_token": rate_value["native_token"],
                "atempo_factor": atempo_factor,
                "execution_status": _NOT_EXECUTED,
            },
            "arousal": arousal_realization,
            "coarse_affect": affect_realization,
        },
        "combination_override": override_meta,
        "validation_status": validation_status,
        "unrepresented_controls": unavailable,
        "warnings": warnings,
        "caveats": [
            (
                "pause_within_count, hesitation_count, and best_effort controls "
                "are not represented by current production canonical v0.2"
            ),
            "turn-boundary pauses remain owned by production assembly",
            "rate postprocessing is planned but has not been executed",
        ],
        "provenance": {
            "source_repository": HIGGS_CONTROL_SOURCE_REPOSITORY,
            "source_commit": HIGGS_CONTROL_SOURCE_COMMIT,
            "contract_identity": contract["mapping_version"],
            "contract_sha256": HIGGS_CONTROL_CONTRACT_SHA256,
        },
    }


def resolve_higgs_turn_controls(turn: Any) -> dict[str, Any]:
    """Resolve the deterministic Higgs plan from a normalized production turn."""
    return resolve_higgs_controls(
        text=turn.text,
        rate=turn.rate,
        arousal=turn.arousal,
        coarse_affect=turn.coarse_affect,
        pause_before_ms=turn.pause_before_ms,
        pause_after_ms=turn.pause_after_ms,
        speaker_id=turn.speaker,
    )
