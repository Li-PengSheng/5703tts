from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from tts5703 import batch_identity
from tts5703.backends.info import backend_static_identity
from tts5703.input.records import InputRecord


def _raw(dialogue_id: str = "A", text: str = "Hello") -> dict:
    return {
        "schema_version": "1.0",
        "dialogue_id": dialogue_id,
        "scenario": {
            "speakers": {
                "caller": {"speaker_id": "C001"},
                "counsellor": {"speaker_id": "L001"},
            }
        },
        "turns": [
            {
                "turn_id": "t1",
                "speaker": "User",
                "text": text,
                "labels": {"opaque": True},
                "acoustic": {
                    "required": {
                        "rate": "normal",
                        "arousal": 2,
                        "affect": "neutral",
                        "pause_before": "none",
                        "pause_within": 0,
                        "hesitations": 0,
                    },
                    "best_effort": {},
                },
            }
        ],
    }


def _record(
    tmp_path: Path,
    *,
    raw: dict | None = None,
    container: str = "corpus.jsonl",
    line: int = 2,
) -> InputRecord:
    value = raw or _raw()
    return InputRecord(
        container_path=tmp_path / container,
        source_format="jsonl",
        line_number=line,
        raw=value,
        dialogue_id=value["dialogue_id"],
    )


def _entry(caller: str = "spk_001", caller_sha: str = "1" * 64) -> dict:
    return {
        "dialogue_id": "A",
        "source": {"line_number": 99, "container_path": "/old/machine/file"},
        "roles": {
            "caller": {
                "upstream_role": "User",
                "upstream_scenario_speaker_id": "C001",
                "render_speaker_id": caller,
                "higgs_reference": {
                    "reference_wav": "refs/caller.wav",
                    "sha256": caller_sha,
                    "resolved_reference_wav": "/machine/local/caller.wav",
                },
                "cosyvoice_reference": {
                    "prompt_wav": "refs/caller-cosy.wav",
                    "prompt_text": "Caller prompt",
                    "sha256": "3" * 64,
                },
            },
            "counsellor": {
                "upstream_role": "Listener",
                "upstream_scenario_speaker_id": "L001",
                "render_speaker_id": "spk_002",
                "higgs_reference": {
                    "reference_wav": "refs/counsellor.wav",
                    "sha256": "2" * 64,
                },
                "cosyvoice_reference": {
                    "prompt_wav": "refs/counsellor-cosy.wav",
                    "prompt_text": "Counsellor prompt",
                    "sha256": "4" * 64,
                },
            },
        },
    }


def _config(engine: str = "higgs") -> dict:
    return {
        "tts": {
            "engine": engine,
            "higgs": {
                "server_executable": "tools/sgl-omni",
                "model_dir": "models/higgs",
                "ffmpeg_bin": "ffmpeg",
            },
            "cosyvoice": {
                "python_bin": "python",
                "repo_dir": "third_party/CosyVoice",
                "model_dir": "models/CosyVoice",
            },
        },
        "fade_ms": 5,
        "telephone": {
            "sample_rate": 8000,
            "channels": 1,
            "high_pass_hz": 300,
            "low_pass_hz": 3400,
            "volume_db_reduction": 2,
        },
    }


def _components(
    record: InputRecord,
    *,
    entry: dict | None = None,
    config: dict | None = None,
) -> dict:
    return batch_identity.render_fingerprint_components(
        record,
        config=config or _config(),
        sidecar_entry=entry or _entry(),
    )


@pytest.mark.parametrize("engine", ["higgs", "cosyvoice"])
def test_batch_identity_is_exact_static_backend_identity(engine: str) -> None:
    identity = backend_static_identity(_config(engine))

    assert batch_identity.backend_identity(_config(engine)) == identity
    assert identity["backend"] == engine
    assert identity["references"] == {}
    assert identity["identity_complete"] is True
    if engine == "higgs":
        assert set(identity) == {
            "backend",
            "model_id",
            "model_dir",
            "server_executable",
            "ffmpeg_bin",
            "host",
            "port",
            "startup_timeout_seconds",
            "inference_timeout_seconds",
            "control_mapping",
            "generation_profile",
            "references",
            "identity_complete",
        }
        assert identity["control_mapping"]["implementation_id"] == (
            "controlled_tts_v1_prod_1"
        )
    else:
        assert set(identity) == {
            "backend",
            "model",
            "model_dir",
            "repo_dir",
            "python_bin",
            "load_trt",
            "load_vllm",
            "fp16",
            "text_frontend",
            "control_mapping",
            "references",
            "identity_complete",
        }
        assert identity["text_frontend"] is False
        assert identity["control_mapping"]["implementation_id"] == (
            "cosyvoice3_final_2"
        )


def test_record_location_and_json_formatting_are_not_fingerprint_identity(
    tmp_path: Path,
) -> None:
    first = _record(tmp_path, container="one/corpus.jsonl", line=2)
    reordered = {key: value for key, value in reversed(list(_raw().items()))}
    moved = _record(
        tmp_path,
        raw=reordered,
        container="different/location.jsonl",
        line=200,
    )

    first_components = _components(first)
    moved_components = _components(moved)

    assert first.record_sha256 == moved.record_sha256
    assert first_components == moved_components
    assert batch_identity.render_fingerprint(
        first_components
    ) == batch_identity.render_fingerprint(moved_components)
    serialized = str(first_components)
    assert "line_number" not in serialized
    assert "container_path" not in serialized
    assert "resolved_reference_wav" not in serialized


def test_selected_reference_declared_path_is_fingerprint_identity(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    first_entry = _entry()
    second_entry = _entry()
    first_entry["roles"]["caller"]["higgs_reference"]["reference_wav"] = (
        "/machine-one/caller.wav"
    )
    second_entry["roles"]["caller"]["higgs_reference"]["reference_wav"] = (
        "/machine-two/caller.wav"
    )

    first = _components(record, entry=first_entry)
    second = _components(record, entry=second_entry)

    assert first != second
    assert (
        first["materialization"]["roles"]["caller"]["higgs_reference"]["reference_wav"]
        == "/machine-one/caller.wav"
    )


@pytest.mark.parametrize(
    ("change", "expected_path"),
    [
        ("source", "source"),
        ("shared_config", "configuration"),
        ("selected_config", "controlled_tts_backend"),
        ("speaker", "materialization"),
        ("reference", "materialization"),
    ],
)
def test_semantic_dependencies_invalidate_fingerprint(
    tmp_path: Path, change: str, expected_path: str
) -> None:
    record = _record(tmp_path)
    baseline = _components(record)
    changed_record = record
    entry = _entry()
    config = _config()
    if change == "source":
        changed_record = _record(tmp_path, raw=_raw(text="Changed"))
    elif change == "shared_config":
        config["fade_ms"] = 7
    elif change == "selected_config":
        config["tts"]["higgs"]["model_dir"] = "models/other-higgs"
    elif change == "speaker":
        entry = _entry(caller="spk_999")
    elif change == "reference":
        entry = _entry(caller_sha="9" * 64)

    changed = _components(
        changed_record,
        entry=entry,
        config=config,
    )

    assert changed[expected_path] != baseline[expected_path]
    assert batch_identity.render_fingerprint(
        changed
    ) != batch_identity.render_fingerprint(baseline)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("implementation_id", "controlled_tts_v1_prod_2"),
        ("contract_sha256", "0" * 64),
        ("generation_profile", {"speed": 0.9}),
    ],
)
def test_backend_semantic_identity_changes_invalidate_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    record = _record(tmp_path)
    baseline = _components(record)
    original = batch_identity._backend_static_identity

    def changed_backend(config: dict) -> dict:
        identity = deepcopy(original(config))
        if field == "implementation_id":
            identity["control_mapping"][field] = value
        elif field == "contract_sha256":
            identity["control_mapping"]["provenance"][field] = value
        else:
            identity[field] = value
        return identity

    monkeypatch.setattr(batch_identity, "_backend_static_identity", changed_backend)
    changed = _components(record)

    assert batch_identity.render_fingerprint(
        changed
    ) != batch_identity.render_fingerprint(baseline)


def test_cosyvoice_identity_freezes_frontend_and_is_fingerprint_identity(
    tmp_path: Path,
) -> None:
    components = _components(_record(tmp_path), config=_config("cosyvoice"))
    identity = components["controlled_tts_backend"]
    changed = deepcopy(components)
    changed["controlled_tts_backend"]["text_frontend"] = True

    assert identity["text_frontend"] is False
    assert identity["control_mapping"]["implementation_id"] == ("cosyvoice3_final_2")
    assert batch_identity.render_fingerprint(
        changed
    ) != batch_identity.render_fingerprint(components)


def test_unrelated_dialogue_state_is_not_a_fingerprint_dependency(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    before = _components(record)

    unrelated_record = _raw("B", "Unrelated change")
    unrelated_sidecar = {"dialogue_id": "B", "roles": deepcopy(_entry()["roles"])}
    unrelated_policy = {
        "schema_version": "1.0",
        "exclusions": [{"dialogue_id": "B", "reason": "Synthetic"}],
    }
    unrelated_assignment_sha = "f" * 64

    assert unrelated_record and unrelated_sidecar and unrelated_policy
    assert unrelated_assignment_sha
    assert _components(record) == before
    assert before["exclusion_decision"] == {"excluded": False}


@pytest.mark.parametrize("engine", ["higgs", "cosyvoice"])
def test_unselected_backend_config_and_reference_are_not_dependencies(
    tmp_path: Path, engine: str
) -> None:
    record = _record(tmp_path)
    config = _config(engine)
    entry = _entry()
    baseline = _components(record, config=config, entry=entry)
    changed_config = deepcopy(config)
    changed_entry = deepcopy(entry)
    unselected = "cosyvoice" if engine == "higgs" else "higgs"
    changed_config["tts"][unselected]["model_dir"] = "models/unselected-change"
    reference = changed_entry["roles"]["caller"][f"{unselected}_reference"]
    reference["sha256"] = "f" * 64
    if unselected == "higgs":
        reference["reference_wav"] = "refs/unselected-other.wav"
    else:
        reference["prompt_wav"] = "refs/unselected-other.wav"
        reference["prompt_text"] = "Changed unselected prompt"

    assert _components(record, config=changed_config, entry=changed_entry) == baseline
