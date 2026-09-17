from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from tts5703 import batch_identity
from tts5703.input_records import InputRecord


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
            },
            "counsellor": {
                "upstream_role": "Listener",
                "upstream_scenario_speaker_id": "L001",
                "render_speaker_id": "spk_002",
                "higgs_reference": {
                    "reference_wav": "refs/counsellor.wav",
                    "sha256": "2" * 64,
                },
            },
        },
    }


def _config() -> dict:
    return {
        "tts": {
            "engine": "higgs",
            "higgs": {
                "server_executable": "tools/sgl-omni",
                "model_dir": "models/higgs",
                "ffmpeg_bin": "ffmpeg",
                "voice_map": {},
            },
        }
    }


def _components(
    record: InputRecord,
    *,
    entry: dict | None = None,
    config_sha: str = "a" * 64,
    registry_sha: str = "b" * 64,
    active_sha: str = "c" * 64,
) -> dict:
    return batch_identity.render_fingerprint_components(
        record,
        config_sha256=config_sha,
        config=_config(),
        sidecar_entry=entry or _entry(),
        registry_sha256=registry_sha,
        active_speakers_sha256=active_sha,
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


def test_machine_local_absolute_reference_path_is_not_fingerprint_identity(
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

    assert first == second
    assert first["materialization"]["roles"]["caller"]["portable_reference_wav"] is None


@pytest.mark.parametrize(
    ("change", "expected_path"),
    [
        ("source", "source"),
        ("config", "configuration"),
        ("speaker", "materialization"),
        ("reference", "materialization"),
        ("registry", "materialization"),
        ("active", "materialization"),
    ],
)
def test_semantic_dependencies_invalidate_fingerprint(
    tmp_path: Path, change: str, expected_path: str
) -> None:
    record = _record(tmp_path)
    baseline = _components(record)
    changed_record = record
    entry = _entry()
    config_sha = "a" * 64
    registry_sha = "b" * 64
    active_sha = "c" * 64
    if change == "source":
        changed_record = _record(tmp_path, raw=_raw(text="Changed"))
    elif change == "config":
        config_sha = "d" * 64
    elif change == "speaker":
        entry = _entry(caller="spk_999")
    elif change == "reference":
        entry = _entry(caller_sha="9" * 64)
    elif change == "registry":
        registry_sha = "8" * 64
    elif change == "active":
        active_sha = "7" * 64

    changed = _components(
        changed_record,
        entry=entry,
        config_sha=config_sha,
        registry_sha=registry_sha,
        active_sha=active_sha,
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
    original = batch_identity.final_controlled_tts_backend_identity

    def changed_backend(config: dict, dialogue: object) -> dict:
        identity = deepcopy(original(config, dialogue))
        if field == "implementation_id":
            identity["control_mapping"][field] = value
        elif field == "contract_sha256":
            identity["control_mapping"]["provenance"][field] = value
        else:
            identity[field] = value
        return identity

    monkeypatch.setattr(
        batch_identity, "final_controlled_tts_backend_identity", changed_backend
    )
    changed = _components(record)

    assert batch_identity.render_fingerprint(
        changed
    ) != batch_identity.render_fingerprint(baseline)


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
