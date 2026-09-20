"""Pinned identity for the pure, CPU-only Higgs control contract.

The contract hash and source provenance make the frozen planner mapping
reviewable independently of the live model. They do not prove reference
conditioning, perceptual control quality, or Cloud runtime compatibility.
"""

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
    """Verify the installed contract bytes, then return an isolated copy."""
    return deepcopy(_contract())
