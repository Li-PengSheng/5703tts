"""Offline tests for the production-owned Controlled TTS v1 Higgs resolver."""

import ast
import hashlib
import inspect
import json
import subprocess
from importlib.resources import files
from pathlib import Path

import pytest

from tts5703 import higgs_controls
from tts5703.higgs_controls import (
    HIGGS_CONTROL_CONTRACT_SHA256,
    HIGGS_CONTROL_SOURCE_COMMIT,
    HIGGS_CONTROL_SOURCE_REPOSITORY,
    HiggsControlError,
    load_higgs_control_contract,
    resolve_higgs_controls,
)

EXPECTED_SHA256 = "4e787a5363d01e5454a8031c59bb1d4bb454ee37d44ccf262d471c7ce30ba799"
EXPECTED_COMMIT = "7b005135e148fad8a142f39dc8be3144c3e5adcc"
UPSTREAM_CONTRACT = Path(
    "comp5703-tts-experiments/mappings/controlled_tts_v1/controlled_tts_v1.json"
)


def _resolve(
    *,
    text: str = "Please stay with me.",
    rate: str = "normal",
    arousal: str | None = "medium",
    coarse_affect: str | None = "neutral",
    pause_before_ms: int = 375,
    pause_after_ms: int = 625,
    speaker_id: str = "spk_001",
) -> dict:
    return resolve_higgs_controls(
        text,
        rate,
        arousal,
        coarse_affect,
        pause_before_ms,
        pause_after_ms,
        speaker_id,
    )


def test_installed_package_contract_loads_and_is_isolated() -> None:
    resource = files("tts5703").joinpath("controlled_tts_v1.json")
    assert resource.is_file()
    first = load_higgs_control_contract()
    first["mapping_version"] = "changed"
    assert load_higgs_control_contract()["mapping_version"] == "controlled_tts_v1"


def test_production_contract_is_byte_exact_upstream_copy_with_verified_hash() -> None:
    production = files("tts5703").joinpath("controlled_tts_v1.json").read_bytes()
    assert hashlib.sha256(production).hexdigest() == EXPECTED_SHA256
    assert HIGGS_CONTROL_CONTRACT_SHA256 == EXPECTED_SHA256
    if UPSTREAM_CONTRACT.is_file():
        assert production == UPSTREAM_CONTRACT.read_bytes()


def test_identity_and_release_status_come_from_contract() -> None:
    contract = load_higgs_control_contract()
    plan = _resolve()
    assert plan["mapping_version"] == contract["mapping_version"]
    assert plan["release_status"] == contract["release_status"]
    assert plan["mapping_version"] == "controlled_tts_v1"
    assert plan["release_status"] == "PROVISIONAL_PRODUCTION_MAPPING_WEEK6_FREEZE"


def test_resolution_is_deterministic() -> None:
    assert _resolve(rate="slow", arousal="high", coarse_affect="sad") == _resolve(
        rate="slow", arousal="high", coarse_affect="sad"
    )


@pytest.mark.parametrize(
    ("rate", "factor", "enabled"),
    [("slow", 0.85, True), ("normal", None, False), ("fast", 1.15, True)],
)
def test_rate_is_unexecuted_atempo_plan_only(
    rate: str, factor: float | None, enabled: bool
) -> None:
    plan = _resolve(rate=rate)
    rate_plan = plan["postprocess"]["rate"]
    realization = plan["realization"]["speaking_rate"]
    assert rate_plan == {
        "processor": "ffmpeg_atempo",
        "enabled": enabled,
        "factor": factor,
        "execution_status": "not_executed",
    }
    assert realization["status"] == "fully_realized"
    assert realization["native_token"] is None
    assert realization["atempo_factor"] == factor
    assert realization["execution_status"] == "not_executed"
    assert plan["higgs"]["native_rate_tokens"] == []


@pytest.mark.parametrize(
    ("arousal", "token", "status"),
    [
        ("low", "<|prosody:expressive_low|>", "best_effort_related_proxy"),
        ("medium", None, "operational_baseline"),
        ("high", "<|prosody:expressive_high|>", "context_dependent_proxy"),
    ],
)
def test_arousal_mapping_matches_contract(
    arousal: str, token: str | None, status: str
) -> None:
    plan = _resolve(arousal=arousal)
    assert plan["realization"]["arousal"] == {
        "requested": arousal,
        "status": status,
        "token": token,
    }
    assert plan["higgs"]["prefix_tokens"] == ([] if token is None else [token])


@pytest.mark.parametrize(
    ("coarse_affect", "token", "status"),
    [
        ("neutral", None, "operational_baseline"),
        ("sad", "<|emotion:helplessness|>", "best_effort_related_proxy"),
        ("anxious", "<|emotion:fear|>", "context_dependent_proxy"),
        ("angry", "<|emotion:anger|>", "fully_realized"),
        ("warm", "<|emotion:affection|>", "fully_realized"),
    ],
)
def test_coarse_affect_mapping_matches_contract(
    coarse_affect: str, token: str | None, status: str
) -> None:
    plan = _resolve(coarse_affect=coarse_affect)
    assert plan["realization"]["coarse_affect"] == {
        "requested": coarse_affect,
        "status": status,
        "token": token,
    }
    assert plan["higgs"]["prefix_tokens"] == ([] if token is None else [token])
    assert "<|emotion:sadness|>" not in plan["higgs"]["model_input"]
    assert "<|emotion:hopelessness|>" not in plan["higgs"]["model_input"]


def test_low_warm_uses_exact_frozen_combination_override() -> None:
    plan = _resolve(arousal="low", coarse_affect="warm")
    assert plan["higgs"]["prefix_tokens"] == ["<|emotion:contentment|>"]
    assert plan["combination_override"] == {
        "id": "low_warm_contentment",
        "matched": {"arousal": "low", "coarse_affect": "warm"},
        "prefix_tokens": ["<|emotion:contentment|>"],
        "status": "best_effort_related_proxy",
        "joint_realization_group": ["arousal.low", "coarse_affect.warm"],
        "replaces": ["arousal.low", "coarse_affect.warm"],
    }
    assert plan["validation_status"] is None
    assert plan["warnings"] == []


def test_absent_arousal_and_affect_are_not_requested() -> None:
    text = "Please stay with me."
    plan = _resolve(text=text, arousal=None, coarse_affect=None)
    not_requested = {
        "requested": None,
        "method": "not_requested",
        "status": None,
        "token": None,
    }
    assert plan["higgs"]["prefix_tokens"] == []
    assert plan["higgs"]["text"] == text
    assert plan["higgs"]["model_input"] == text
    assert plan["requested"]["arousal"] is None
    assert plan["requested"]["coarse_affect"] is None
    assert plan["normalized"]["arousal"] is None
    assert plan["normalized"]["coarse_affect"] is None
    assert plan["realization"]["arousal"] == not_requested
    assert plan["realization"]["coarse_affect"] == not_requested
    assert plan["combination_override"] is None
    assert plan["validation_status"] is None
    assert plan["warnings"] == []


def test_absent_arousal_with_sad_affect_emits_only_helplessness() -> None:
    plan = _resolve(arousal=None, coarse_affect="sad")
    assert plan["higgs"]["prefix_tokens"] == ["<|emotion:helplessness|>"]
    assert plan["requested"]["arousal"] is None
    assert plan["realization"]["arousal"]["method"] == "not_requested"
    assert plan["realization"]["arousal"]["status"] is None
    assert plan["realization"]["arousal"]["token"] is None
    assert plan["combination_override"] is None
    assert plan["validation_status"] is None
    assert plan["warnings"] == []


def test_low_arousal_with_absent_affect_emits_only_expressive_low() -> None:
    plan = _resolve(arousal="low", coarse_affect=None)
    assert plan["higgs"]["prefix_tokens"] == ["<|prosody:expressive_low|>"]
    assert plan["requested"]["coarse_affect"] is None
    assert plan["realization"]["coarse_affect"]["method"] == "not_requested"
    assert plan["realization"]["coarse_affect"]["status"] is None
    assert plan["realization"]["coarse_affect"]["token"] is None
    assert plan["combination_override"] is None
    assert plan["validation_status"] is None
    assert plan["warnings"] == []


def test_model_input_is_prefix_plus_unmodified_production_text() -> None:
    text = "I tried to explain. Then I stopped."
    plan = _resolve(text=text, arousal="high", coarse_affect="angry")
    assert plan["higgs"]["text"] == text
    assert plan["higgs"]["model_input"] == (
        "<|prosody:expressive_high|><|emotion:anger|>" + text
    )
    assert plan["higgs"]["synthesis_call_count"] == 1
    assert "ummm" not in plan["higgs"]["text"]
    assert "<|prosody:pause|>" not in plan["higgs"]["text"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rate", "very_fast"),
        ("rate", None),
        ("arousal", "frantic"),
        ("arousal", 2),
        ("coarse_affect", "distressed"),
    ],
)
def test_invalid_required_domains_are_rejected(field: str, value: object) -> None:
    arguments = {field: value}
    with pytest.raises(HiggsControlError, match=rf"Unsupported Higgs {field} mapping"):
        _resolve(**arguments)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pause_before_ms", -1),
        ("pause_after_ms", True),
        ("pause_after_ms", 1.5),
    ],
)
def test_invalid_production_pause_values_are_rejected(
    field: str, value: object
) -> None:
    with pytest.raises(HiggsControlError, match=field):
        _resolve(**{field: value})


def test_no_reference_asset_or_embedding_is_invented() -> None:
    plan = _resolve()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                assert "wav" not in key.lower()
                assert "embedding" not in key.lower()
                assert "reference" not in key.lower()
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(plan)


def test_resolution_launches_no_process(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("control resolution must not launch a process")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    assert _resolve()["backend"] == "higgs"


def test_numeric_turn_boundary_pauses_remain_pipeline_owned() -> None:
    plan = _resolve(pause_before_ms=373, pause_after_ms=947)
    pauses = plan["postprocess"]["turn_boundary_pauses"]
    assert pauses == {
        "owner": "pipeline_assembly",
        "pause_before_ms": 373,
        "pause_after_ms": 947,
        "audio_insertion_planned": False,
    }
    assert set(plan["postprocess"]) == {"rate", "turn_boundary_pauses"}


def test_unrepresented_controls_are_explicitly_not_realized() -> None:
    unavailable = _resolve()["unrepresented_controls"]
    assert set(unavailable) == {
        "pause_within_count",
        "hesitation_count",
        "best_effort",
    }
    assert all(
        item
        == {
            "availability": "not_represented_by_current_canonical",
            "realization": None,
        }
        for item in unavailable.values()
    )


def test_provenance_pins_upstream_repository_commit_and_contract() -> None:
    provenance = _resolve()["provenance"]
    assert HIGGS_CONTROL_SOURCE_REPOSITORY == "ginkgoyin/comp5703-tts-experiments"
    assert HIGGS_CONTROL_SOURCE_COMMIT == EXPECTED_COMMIT
    assert provenance == {
        "source_repository": "ginkgoyin/comp5703-tts-experiments",
        "source_commit": EXPECTED_COMMIT,
        "contract_identity": "controlled_tts_v1",
        "contract_sha256": EXPECTED_SHA256,
    }


def test_runtime_module_has_no_experiment_import() -> None:
    tree = ast.parse(inspect.getsource(higgs_controls))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(
        name.startswith(("mappings", "comp5703_tts_experiments")) for name in imported
    )


def test_runtime_constants_and_resolutions_agree_with_frozen_json() -> None:
    contract = json.loads(
        files("tts5703").joinpath("controlled_tts_v1.json").read_text()
    )
    for rate in contract["controls"]["speaking_rate"]["domains"]:
        value = contract["controls"]["speaking_rate"]["values"][rate]
        plan = _resolve(rate=rate)
        assert (
            plan["realization"]["speaking_rate"]["atempo_factor"]
            == value["atempo_factor"]
        )
        assert plan["realization"]["speaking_rate"]["status"] == value["status"]
    for arousal, value in contract["controls"]["arousal"]["values"].items():
        assert (
            _resolve(arousal=arousal)["realization"]["arousal"]["token"]
            == value["token"]
        )
    for affect, value in contract["controls"]["coarse_affect"]["values"].items():
        assert (
            _resolve(coarse_affect=affect)["realization"]["coarse_affect"]["token"]
            == value["token"]
        )
