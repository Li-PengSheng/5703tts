"""Load the frozen mapping contract and normalize final source turns.

Normalization is source semantics, not backend execution planning.  It requires
rate, arousal, affect, pause_before, pause_within, and hesitations; recognizes
best-effort affect_fine, volume, flattened_affect, and events without claiming
that a backend realizes them; and maps upstream ``User``/``Listener`` to logical
``caller``/``counsellor`` while preserving the scenario speaker ID.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
from typing import Any

CANONICAL_STATUS_VOCABULARY = (
    "fully_realized",
    "operational_baseline",
    "best_effort_related_proxy",
    "context_dependent_proxy",
    "jointly_realized",
    "unsupported",
    "not_realized",
    "diagnostic_only",
    "unvalidated_combination",
)


class UpstreamValidationError(ValueError):
    def __init__(self, path: str, value: Any, expected: str) -> None:
        self.path, self.value, self.expected = path, value, expected
        super().__init__(f"Invalid {path}: {value!r}; expected {expected}")


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key in Controlled TTS v1 JSON: {key!r}")
        result[key] = value
    return result


def _validate_statuses(
    value: Any, vocabulary: set[str], path: str = "contract"
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if (
                key
                in {
                    "status",
                    "positive_status",
                    "zero_status",
                    "validation_status",
                    "validation_status_when_unlisted",
                }
                and item is not None
                and item not in vocabulary
            ):
                raise ValueError(
                    f"Invalid Controlled TTS v1 status at {item_path}: {item!r}"
                )
            _validate_statuses(item, vocabulary, item_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_statuses(item, vocabulary, f"{path}[{index}]")


@lru_cache(maxsize=1)
def _contract() -> dict[str, Any]:
    data = files("tts5703").joinpath("controlled_tts_v1.json").read_bytes()
    contract = json.loads(data, object_pairs_hook=_no_duplicate_object)
    required = {
        "schema_version",
        "mapping_version",
        "release_status",
        "model",
        "reference_route",
        "status_vocabulary",
        "controls",
        "combination_overrides",
        "planner_contracts",
        "upstream_normalization",
        "best_effort_policy",
        "unseen_combination_policy",
        "forbidden_realizations",
        "composition_order",
        "evidence",
    }
    missing = required - set(contract)
    if (
        missing
        or contract["mapping_version"] != "controlled_tts_v1"
        or contract["release_status"] != "PROVISIONAL_PRODUCTION_MAPPING_WEEK6_FREEZE"
    ):
        raise ValueError(
            f"Invalid Controlled TTS v1 contract: missing={sorted(missing)}"
        )
    if tuple(contract["status_vocabulary"]) != CANONICAL_STATUS_VOCABULARY:
        raise ValueError("Invalid Controlled TTS v1 status vocabulary")
    vocabulary = set(contract["status_vocabulary"])
    controls = contract["controls"]
    arousal = controls.get("arousal", {})
    if set(arousal.get("upstream_values", {})) != {"1", "2", "3"} or set(
        arousal.get("upstream_values", {}).values()
    ) != set(arousal.get("values", {})):
        raise ValueError("Invalid Controlled TTS v1 arousal definitions")
    if controls.get("hesitation_count", {}).get("domains") != [0, 1, 2]:
        raise ValueError("Invalid Controlled TTS v1 hesitation domain")
    hesitation_events = controls["hesitation_count"]["values"]["1"]["events"]
    pause_token = controls["pause_within_count"].get("token")
    diagnostic_p9 = (
        hesitation_events[0] + pause_token
        if len(hesitation_events) == 1 and isinstance(pause_token, str)
        else None
    )
    if (
        len(contract["forbidden_realizations"])
        != len(set(contract["forbidden_realizations"]))
        or diagnostic_p9 not in contract["forbidden_realizations"]
    ):
        raise ValueError("Invalid Controlled TTS v1 forbidden realization policy")
    if contract["model"].get("synthesis_call_count") != 1:
        raise ValueError("Invalid Controlled TTS v1 synthesis-call contract")
    matches = [
        tuple(sorted(item.get("match", {}).items()))
        for item in contract["combination_overrides"]
    ]
    identifiers = [item.get("id") for item in contract["combination_overrides"]]
    if (
        len(matches) != len(set(matches))
        or len(identifiers) != len(set(identifiers))
        or any(not item for item in identifiers)
    ):
        raise ValueError("Duplicate or invalid Controlled TTS v1 combination override")
    policy_statuses = contract["best_effort_policy"].get("statuses", {})
    if set(policy_statuses) != {
        "jointly_realized",
        "empty_or_absent",
        "unsupported",
    } or any(value not in vocabulary for value in policy_statuses.values()):
        raise ValueError("Invalid Controlled TTS v1 best-effort status policy")
    _validate_statuses(contract, vocabulary)
    return contract


def load_contract() -> dict[str, Any]:
    """Return an isolated copy of the canonical contract."""
    return deepcopy(_contract())


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise UpstreamValidationError(path, value, "object")
    return value


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise UpstreamValidationError(f"{path}.{key}", None, "present value")
    return mapping[key]


def _domain(value: Any, allowed: list[Any], path: str) -> Any:
    if value not in allowed:
        raise UpstreamValidationError(path, value, "one of " + repr(allowed))
    return value


def _nonnegative_int(value: Any, path: str, *, allowed: list[int] | None = None) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or (allowed is not None and value not in allowed)
    ):
        expectation = (
            f"one of {allowed}" if allowed is not None else "non-negative integer"
        )
        raise UpstreamValidationError(path, value, expectation)
    return value


def normalize_turn(
    turn: Mapping[str, Any],
    *,
    dialogue_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one final-schema source turn and return backend-neutral values.

    When dialogue context is supplied, the returned ``speaker_id`` is the
    upstream scenario identity.  Production ``spk_*`` assignment belongs to the
    speaker sidecar and is introduced later during canonicalization.
    """
    contract = _contract()
    turn = _mapping(turn, "turn")
    text = _required(turn, "text", "turn")
    if not isinstance(text, str) or not text.strip():
        raise UpstreamValidationError("turn.text", text, "nonblank string")
    upstream_role = _required(turn, "speaker", "turn")
    roles = contract["upstream_normalization"]["roles"]
    if upstream_role not in roles:
        raise UpstreamValidationError("turn.speaker", upstream_role, "User or Listener")
    acoustic = _mapping(_required(turn, "acoustic", "turn"), "turn.acoustic")
    required = _mapping(
        _required(acoustic, "required", "turn.acoustic"), "turn.acoustic.required"
    )
    controls = contract["controls"]
    rate = _domain(
        _required(required, "rate", "turn.acoustic.required"),
        controls["speaking_rate"]["domains"],
        "turn.acoustic.required.rate",
    )
    arousal_values = controls["arousal"]["upstream_values"]
    raw_arousal = _nonnegative_int(
        _required(required, "arousal", "turn.acoustic.required"),
        "turn.acoustic.required.arousal",
        allowed=sorted(int(value) for value in arousal_values),
    )
    affect = _domain(
        _required(required, "affect", "turn.acoustic.required"),
        controls["coarse_affect"]["domains"],
        "turn.acoustic.required.affect",
    )
    pause_before = _domain(
        _required(required, "pause_before", "turn.acoustic.required"),
        controls["pause_before"]["domains"],
        "turn.acoustic.required.pause_before",
    )
    within = _nonnegative_int(
        _required(required, "pause_within", "turn.acoustic.required"),
        "turn.acoustic.required.pause_within",
    )
    hesitations = _nonnegative_int(
        _required(required, "hesitations", "turn.acoustic.required"),
        "turn.acoustic.required.hesitations",
        allowed=controls["hesitation_count"]["domains"],
    )
    best_effort_raw = acoustic.get("best_effort")
    best_effort = dict(best_effort_raw) if isinstance(best_effort_raw, Mapping) else {}
    expected_best_effort = contract["upstream_normalization"]["best_effort_fields"]
    missing_best_effort = (
        [] if isinstance(best_effort_raw, Mapping) else list(expected_best_effort)
    )
    if isinstance(best_effort_raw, Mapping):
        missing_best_effort = [
            key for key in expected_best_effort if key not in best_effort
        ]
    logical = roles[upstream_role]
    speaker_id = None
    dialogue_id = None
    provenance: dict[str, Any] = {}
    if dialogue_context is not None:
        context = _mapping(dialogue_context, "dialogue_context")
        dialogue_id = context.get("dialogue_id")
        scenario = _mapping(
            _required(context, "scenario", "dialogue_context"),
            "dialogue_context.scenario",
        )
        speakers = _mapping(
            _required(scenario, "speakers", "dialogue_context.scenario"),
            "dialogue_context.scenario.speakers",
        )
        role_spec = _mapping(
            _required(speakers, logical, "dialogue_context.scenario.speakers"),
            f"dialogue_context.scenario.speakers.{logical}",
        )
        speaker_id = _required(
            role_spec,
            "speaker_id",
            f"dialogue_context.scenario.speakers.{logical}",
        )
        if not isinstance(speaker_id, str) or not speaker_id:
            raise UpstreamValidationError(
                "dialogue_context.scenario.speakers.*.speaker_id",
                speaker_id,
                "nonempty string",
            )
        if "known_issue" in context:
            provenance["known_issue"] = deepcopy(context["known_issue"])
    return {
        "text": text,
        "turn_id": turn.get("turn_id"),
        "dialogue_id": dialogue_id,
        "upstream_role": upstream_role,
        "role": logical,
        "speaker_id": speaker_id,
        "required_raw": deepcopy(dict(required)),
        "best_effort": deepcopy(best_effort),
        "missing_best_effort": missing_best_effort,
        "provenance": provenance,
        "rate": rate,
        "arousal": arousal_values[str(raw_arousal)],
        "coarse_affect": affect,
        "pause_before": pause_before,
        "pause_within_count": within,
        "hesitation_count": hesitations,
    }
