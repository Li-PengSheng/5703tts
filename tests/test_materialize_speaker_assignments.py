"""Offline tests for speaker-assignment materialization. No TTS or GPU."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from tts5703.config import load_config
from tts5703.validate import load_and_validate

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "materialize_speaker_assignments.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "materialize_speaker_assignments", SCRIPT_PATH
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
materialize = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = materialize
SCRIPT_SPEC.loader.exec_module(materialize)

ACTIVE = (
    "spk_001",
    "spk_003",
    "spk_006",
    "spk_009",
)
EXCLUDED = "spk_002"
PREFIX = "You are a helpful assistant.<|endofprompt|>"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, payload: dict[str, Any] | str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    elif isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def _dialogue(
    dialogue_id: str,
    *,
    caller_text: str = "I needed to talk.",
    counsellor_text: str = "I am here to listen.",
    rate: str = "normal",
    arousal: str | None = None,
    coarse_affect: str | None = None,
    extra_role: str | None = None,
) -> dict[str, Any]:
    turns = [
        {
            "turn_id": 1,
            "speaker": "counsellor",
            "text": counsellor_text,
            "label": "normal",
            "acoustic_spec": {
                "rate": rate,
                "pause_before_ms": 0,
                "pause_after_ms": 400,
                "arousal": arousal,
                "coarse_affect": coarse_affect,
                "emotion": None,
                "paralinguistic_events": [],
            },
        },
        {
            "turn_id": 2,
            "speaker": "caller",
            "text": caller_text,
            "label": "alert",
            "acoustic_spec": {
                "rate": "slow",
                "pause_before_ms": 0,
                "pause_after_ms": 800,
                "arousal": arousal,
                "coarse_affect": coarse_affect,
                "emotion": None,
                "paralinguistic_events": [],
            },
        },
    ]
    if extra_role is not None:
        turns.append(
            {
                "turn_id": 3,
                "speaker": extra_role,
                "text": "I can stay on the line.",
                "label": "confirm",
                "acoustic_spec": {
                    "rate": "normal",
                    "pause_before_ms": 0,
                    "pause_after_ms": 200,
                    "arousal": None,
                    "coarse_affect": None,
                    "emotion": None,
                    "paralinguistic_events": [],
                },
            }
        )
    return {
        "schema_version": "0.2",
        "dialogue_id": dialogue_id,
        "turns": turns,
    }


def _final_dialogue(dialogue_id: str = "final_a") -> dict[str, Any]:
    required = {
        "rate": "normal",
        "arousal": 2,
        "affect": "neutral",
        "pause_before": "none",
        "pause_within": 1,
        "hesitations": 1,
    }
    return {
        "schema_version": "1.0",
        "dialogue_id": dialogue_id,
        "scenario": {
            "speakers": {
                "caller": {"speaker_id": "C123"},
                "counsellor": {"speaker_id": "L456"},
            }
        },
        "turns": [
            {
                "turn_id": 1,
                "speaker": "User",
                "text": "I need help.",
                "labels": {"upstream": ["opaque"]},
                "acoustic": {"required": required, "best_effort": {}},
            },
            {
                "turn_id": 2,
                "speaker": "Listener",
                "text": "I am here.",
                "labels": ["opaque", "plural"],
                "acoustic": {"required": {**required, "pause_within": 0}},
            },
        ],
    }


def _wav(path: Path, payload: bytes) -> str:
    _write(path, payload)
    return _sha256_bytes(payload)


def _pool(tmp_path: Path) -> dict[str, Path]:
    wavs = tmp_path / "refs"
    hashes = {}
    for speaker_id, blob in {
        "spk_001": b"WAV001",
        "spk_002": b"WAV002",
        "spk_003": b"WAV003",
        "spk_006": b"WAV006",
        "spk_009": b"WAV009",
    }.items():
        hashes[speaker_id] = _wav(wavs / f"{speaker_id}.wav", blob)
    registry = {
        "speaker_pool_version": "vctk_v0.1",
        "speakers": [
            {
                "speaker_id": speaker_id,
                "source_speaker_id": f"p{speaker_id[-3:]}",
                "primary_reference": {
                    "prompt_wav": f"refs/{speaker_id}.wav",
                    "prompt_text": f"transcript for {speaker_id}",
                    "sha256": hashes[speaker_id],
                },
            }
            for speaker_id in ("spk_001", "spk_002", "spk_003", "spk_006", "spk_009")
        ],
    }
    active = {
        "speaker_pool_version": "vctk_v0.1",
        "status": "active_pool_v0.1",
        "selection_basis": "test",
        "active_speaker_ids": list(ACTIVE),
        "excluded_from_active_pool": [
            {"speaker_id": EXCLUDED, "reason": "uncontrolled pause"}
        ],
    }
    # Four active speakers so the v0.1 exact-count check is not used: the
    # assignment helper only enforces 10 when version is vctk_v0.1. Use a
    # test version to keep the fixture small.
    registry["speaker_pool_version"] = "test_pool"
    active["speaker_pool_version"] = "test_pool"
    base_config = {
        "tts": {
            "engine": "cosyvoice",
            "default_rate": "+0%",
            "cosyvoice": {
                "python_bin": "third_party/CosyVoice/.venv/bin/python",
                "repo_dir": "third_party/CosyVoice",
                "model_dir": "models/Fun-CosyVoice3-0.5B",
                "load_trt": False,
                "load_vllm": False,
                "fp16": True,
                "voice_map": {
                    "placeholder": {
                        "prompt_wav": "refs/spk_001.wav",
                        "prompt_text": "placeholder",
                    }
                },
            },
        },
        "pause": {"default_ms": 500},
        "fade_ms": 5,
        "telephone": {
            "sample_rate": 8000,
            "channels": 1,
            "high_pass_hz": 300,
            "low_pass_hz": 3400,
            "volume_db_reduction": 3,
        },
        "keep_me": "unrelated",
    }
    return {
        "registry": _write(tmp_path / "speaker_registry.json", registry),
        "active": _write(tmp_path / "active_speakers.json", active),
        "base_config": _write(
            tmp_path / "base_config.yaml",
            yaml.safe_dump(base_config, sort_keys=False),
        ),
        "root": tmp_path,
    }


def _assignments(path: Path, rows: list[dict[str, Any]]) -> Path:
    text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(", ", ": ")) + "\n"
        for row in rows
    )
    return _write(path, text)


def _approve_higgs(pool: dict[str, Path], speaker_ids: tuple[str, ...]) -> None:
    registry = json.loads(pool["registry"].read_text(encoding="utf-8"))
    by_id = {entry["speaker_id"]: entry for entry in registry["speakers"]}
    for speaker_id in speaker_ids:
        reference = f"higgs_refs/{speaker_id}.wav"
        payload = f"HIGGS-{speaker_id}".encode()
        _write(pool["root"] / reference, payload)
        by_id[speaker_id]["higgs_reference"] = {
            "reference_wav": reference,
            "sha256": _sha256_bytes(payload),
        }
    _write(pool["registry"], registry)

    config = yaml.safe_load(pool["base_config"].read_text(encoding="utf-8"))
    config["tts"]["higgs"] = {
        "server_executable": "/opt/higgs/bin/sgl-omni",
        "model_dir": "/opt/higgs/model",
        "voice_map": {"placeholder": {"reference_wav": "unused.wav"}},
    }
    _write(pool["base_config"], yaml.safe_dump(config, sort_keys=False))


def _run(
    tmp_path: Path,
    pool: dict[str, Path],
    dialogues: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    output_name: str = "render_input",
    higgs_ready: bool = False,
) -> dict[str, Any]:
    input_dir = tmp_path / "canonical"
    for filename, payload in dialogues.items():
        _write(input_dir / filename, payload)
    assignments = _assignments(tmp_path / "speaker_assignments.jsonl", rows)
    output_dir = tmp_path / output_name
    config_out = tmp_path / "materialized_config.yaml"
    return materialize.materialize_speaker_assignments(
        input_path=input_dir,
        assignments_path=assignments,
        registry_path=pool["registry"],
        active_speakers_path=pool["active"],
        output_dir=output_dir,
        config_out=config_out,
        base_config_path=pool["base_config"],
        project_root=pool["root"],
        higgs_ready=higgs_ready,
    )


def test_two_dialogue_materialization_rewrites_speakers_and_voice_map(
    tmp_path: Path,
) -> None:
    pool = _pool(tmp_path)
    dialogues = {
        "a.json": _dialogue("dialogue_a", caller_text="Hello from A."),
        "b.json": _dialogue("dialogue_b", caller_text="Hello from B."),
    }
    rows = [
        {
            "dialogue_id": "dialogue_a",
            "speaker_pool_version": "test_pool",
            "assignment_seed": 5703,
            "role_assignments": {"counsellor": "spk_003", "caller": "spk_001"},
        },
        {
            "dialogue_id": "dialogue_b",
            "speaker_pool_version": "test_pool",
            "assignment_seed": 5703,
            "role_assignments": {"counsellor": "spk_009", "caller": "spk_006"},
        },
    ]

    manifest = _run(tmp_path, pool, dialogues, rows)
    rendered_a = json.loads((tmp_path / "render_input" / "a.json").read_text())
    rendered_b = json.loads((tmp_path / "render_input" / "b.json").read_text())
    config = load_config(tmp_path / "materialized_config.yaml")

    assert rendered_a["turns"][0]["speaker"] == "spk_003"
    assert rendered_a["turns"][1]["speaker"] == "spk_001"
    assert rendered_b["turns"][0]["speaker"] == "spk_009"
    assert rendered_b["turns"][1]["speaker"] == "spk_006"
    assert rendered_a["turns"][1]["text"] == "Hello from A."
    assert rendered_b["turns"][1]["text"] == "Hello from B."

    voice_map = config["tts"]["cosyvoice"]["voice_map"]
    assert set(voice_map) == {"spk_001", "spk_003", "spk_006", "spk_009"}
    assert voice_map["spk_001"]["prompt_wav"] == "refs/spk_001.wav"
    assert voice_map["spk_001"]["prompt_text"] == f"{PREFIX}transcript for spk_001"
    assert config["keep_me"] == "unrelated"
    assert "higgs" not in config["tts"]

    load_and_validate(tmp_path / "render_input" / "a.json", config)
    load_and_validate(tmp_path / "render_input" / "b.json", config)
    assert manifest["input_count"] == 2
    assert "placeholder" not in voice_map


def test_higgs_ready_materialization_uses_only_explicit_approved_references(
    tmp_path: Path,
) -> None:
    pool = _pool(tmp_path)
    _approve_higgs(pool, ("spk_001", "spk_003"))
    _run(
        tmp_path,
        pool,
        {"a.json": _dialogue("dialogue_a")},
        [
            {
                "dialogue_id": "dialogue_a",
                "role_assignments": {
                    "counsellor": "spk_003",
                    "caller": "spk_001",
                },
            }
        ],
        higgs_ready=True,
    )
    config = yaml.safe_load((tmp_path / "materialized_config.yaml").read_text())
    cosyvoice_map = config["tts"]["cosyvoice"]["voice_map"]
    higgs_map = config["tts"]["higgs"]["voice_map"]

    assert set(higgs_map) == set(cosyvoice_map) == {"spk_001", "spk_003"}
    assert higgs_map == {
        "spk_001": {"reference_wav": "higgs_refs/spk_001.wav"},
        "spk_003": {"reference_wav": "higgs_refs/spk_003.wav"},
    }
    assert cosyvoice_map["spk_001"]["prompt_wav"] == "refs/spk_001.wav"
    assert all(set(voice) == {"reference_wav"} for voice in higgs_map.values())


def test_higgs_ready_never_reuses_primary_reference_implicitly(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    with pytest.raises(
        materialize.SpeakerMaterializationError,
        match="spk_003 has no approved higgs_reference",
    ):
        _run(
            tmp_path,
            pool,
            {"a.json": _dialogue("dialogue_a")},
            [
                {
                    "dialogue_id": "dialogue_a",
                    "role_assignments": {
                        "counsellor": "spk_003",
                        "caller": "spk_001",
                    },
                }
            ],
            higgs_ready=True,
        )


def test_final_higgs_only_materialization_writes_sidecar_without_rewriting_source(
    tmp_path: Path,
) -> None:
    pool = _pool(tmp_path)
    _approve_higgs(pool, ("spk_001", "spk_003"))
    registry = json.loads(pool["registry"].read_text(encoding="utf-8"))
    for entry in registry["speakers"]:
        if entry["speaker_id"] in {"spk_001", "spk_003"}:
            entry.pop("primary_reference")
    _write(pool["registry"], registry)
    base_config = yaml.safe_load(pool["base_config"].read_text(encoding="utf-8"))
    base_config["tts"].pop("cosyvoice")
    base_config["tts"]["engine"] = "higgs"
    _write(pool["base_config"], yaml.safe_dump(base_config, sort_keys=False))

    source = tmp_path / "corpus.jsonl"
    source.write_text("\n" + json.dumps(_final_dialogue()) + "\n", encoding="utf-8")
    source_before = source.read_bytes()
    assignments = _assignments(
        tmp_path / "speaker_assignments.jsonl",
        [
            {
                "dialogue_id": "final_a",
                "role_assignments": {
                    "caller": "spk_001",
                    "counsellor": "spk_003",
                },
            }
        ],
    )

    manifest = materialize.materialize_speaker_assignments(
        input_path=source,
        assignments_path=assignments,
        registry_path=pool["registry"],
        active_speakers_path=pool["active"],
        output_dir=tmp_path / "render_input",
        config_out=tmp_path / "higgs_config.yaml",
        base_config_path=pool["base_config"],
        project_root=pool["root"],
        higgs_ready=True,
    )

    assert source.read_bytes() == source_before
    parsed_source = json.loads(source.read_text(encoding="utf-8").strip())
    assert [turn["speaker"] for turn in parsed_source["turns"]] == [
        "User",
        "Listener",
    ]
    assert not any((tmp_path / "render_input").iterdir())
    assert manifest["materialization_mode"] == "final_speaker_sidecar"
    assert manifest["source_records_rewritten"] is False
    assert manifest["registry_sha256"] == _sha256_bytes(pool["registry"].read_bytes())
    assert manifest["active_speakers_sha256"] == _sha256_bytes(
        pool["active"].read_bytes()
    )
    row = manifest["dialogues"][0]
    assert row["source"]["container_path"] == str(source)
    assert row["source"]["format"] == "jsonl"
    assert row["source"]["line_number"] == 2
    assert len(row["source"]["record_sha256"]) == 64
    assert row["roles"] == {
        "caller": {
            "upstream_role": "User",
            "upstream_scenario_speaker_id": "C123",
            "render_speaker_id": "spk_001",
            "higgs_reference": {
                "reference_wav": "higgs_refs/spk_001.wav",
                "sha256": _sha256_bytes(b"HIGGS-spk_001"),
            },
        },
        "counsellor": {
            "upstream_role": "Listener",
            "upstream_scenario_speaker_id": "L456",
            "render_speaker_id": "spk_003",
            "higgs_reference": {
                "reference_wav": "higgs_refs/spk_003.wav",
                "sha256": _sha256_bytes(b"HIGGS-spk_003"),
            },
        },
    }
    config = yaml.safe_load((tmp_path / "higgs_config.yaml").read_text())
    assert "cosyvoice" not in config["tts"]
    assert config["tts"]["higgs"]["voice_map"] == {
        "spk_001": {"reference_wav": "higgs_refs/spk_001.wav"},
        "spk_003": {"reference_wav": "higgs_refs/spk_003.wav"},
    }
    assert "C123" not in config["tts"]["higgs"]["voice_map"]
    assert "L456" not in config["tts"]["higgs"]["voice_map"]


def test_final_sidecar_file_hashes_track_registry_and_active_pool_bytes(
    tmp_path: Path,
) -> None:
    pool = _pool(tmp_path)
    dialogues = {"final.json": _final_dialogue()}
    rows = [
        {
            "dialogue_id": "final_a",
            "role_assignments": {
                "caller": "spk_001",
                "counsellor": "spk_003",
            },
        }
    ]

    first = _run(tmp_path, pool, dialogues, rows, output_name="sidecar_one")
    assert first["registry_sha256"] == _sha256_bytes(pool["registry"].read_bytes())
    assert first["active_speakers_sha256"] == _sha256_bytes(pool["active"].read_bytes())
    caller = first["dialogues"][0]["roles"]["caller"]
    assert caller["cosyvoice_reference"] == {
        "prompt_wav": "refs/spk_001.wav",
        "prompt_text": (
            "You are a helpful assistant.<|endofprompt|>transcript for spk_001"
        ),
        "sha256": _sha256_bytes(b"WAV001"),
    }
    assert "higgs_reference" not in caller

    pool["registry"].write_text(
        pool["registry"].read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    second = _run(tmp_path, pool, dialogues, rows, output_name="sidecar_two")
    assert second["registry_sha256"] != first["registry_sha256"]
    assert second["active_speakers_sha256"] == first["active_speakers_sha256"]

    pool["active"].write_text(
        pool["active"].read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    third = _run(tmp_path, pool, dialogues, rows, output_name="sidecar_three")
    assert third["registry_sha256"] == second["registry_sha256"]
    assert third["active_speakers_sha256"] != second["active_speakers_sha256"]


def test_final_sidecar_carries_separate_references_for_both_backends(
    tmp_path: Path,
) -> None:
    pool = _pool(tmp_path)
    _approve_higgs(pool, ("spk_001", "spk_003"))

    manifest = _run(
        tmp_path,
        pool,
        {"final.json": _final_dialogue()},
        [
            {
                "dialogue_id": "final_a",
                "role_assignments": {
                    "caller": "spk_001",
                    "counsellor": "spk_003",
                },
            }
        ],
        higgs_ready=True,
    )

    caller = manifest["dialogues"][0]["roles"]["caller"]
    assert caller["cosyvoice_reference"]["prompt_wav"] == "refs/spk_001.wav"
    assert caller["cosyvoice_reference"]["prompt_text"].endswith(
        "transcript for spk_001"
    )
    assert caller["higgs_reference"]["reference_wav"] == ("higgs_refs/spk_001.wav")
    assert (
        caller["cosyvoice_reference"]["sha256"] != caller["higgs_reference"]["sha256"]
    )


def test_final_higgs_ready_requires_approved_reference_not_primary(
    tmp_path: Path,
) -> None:
    pool = _pool(tmp_path)
    _approve_higgs(pool, ())
    base_config = yaml.safe_load(pool["base_config"].read_text(encoding="utf-8"))
    base_config["tts"].pop("cosyvoice")
    _write(pool["base_config"], yaml.safe_dump(base_config, sort_keys=False))
    source = _write(tmp_path / "final.json", _final_dialogue())
    assignments = _assignments(
        tmp_path / "speaker_assignments.jsonl",
        [
            {
                "dialogue_id": "final_a",
                "role_assignments": {
                    "caller": "spk_001",
                    "counsellor": "spk_003",
                },
            }
        ],
    )

    with pytest.raises(
        materialize.SpeakerMaterializationError,
        match="spk_001 has no approved higgs_reference",
    ):
        materialize.materialize_speaker_assignments(
            input_path=source,
            assignments_path=assignments,
            registry_path=pool["registry"],
            active_speakers_path=pool["active"],
            output_dir=tmp_path / "render_input",
            config_out=tmp_path / "config.yaml",
            base_config_path=pool["base_config"],
            project_root=pool["root"],
            higgs_ready=True,
        )
    assert not (tmp_path / "render_input").exists()
    assert not (tmp_path / "config.yaml").exists()


def test_mixed_final_and_rewrite_families_fail_before_outputs(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    input_dir = tmp_path / "canonical"
    _write(input_dir / "legacy.json", _dialogue("old"))
    _write(input_dir / "final.json", _final_dialogue("new"))
    assignments = _assignments(
        tmp_path / "speaker_assignments.jsonl",
        [
            {
                "dialogue_id": "old",
                "role_assignments": {
                    "caller": "spk_001",
                    "counsellor": "spk_003",
                },
            },
            {
                "dialogue_id": "new",
                "role_assignments": {
                    "caller": "spk_006",
                    "counsellor": "spk_009",
                },
            },
        ],
    )

    with pytest.raises(
        materialize.SpeakerMaterializationError,
        match="Cannot materialize final nested and legacy/v0.2",
    ):
        materialize.materialize_speaker_assignments(
            input_path=input_dir,
            assignments_path=assignments,
            registry_path=pool["registry"],
            active_speakers_path=pool["active"],
            output_dir=tmp_path / "render_input",
            config_out=tmp_path / "config.yaml",
            base_config_path=pool["base_config"],
            manifest_path=tmp_path / "manifest.json",
            project_root=pool["root"],
        )
    assert not (tmp_path / "render_input").exists()
    assert not (tmp_path / "config.yaml").exists()
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("invalid_hash", [None, "", "0" * 63, "g" * 64])
def test_higgs_reference_requires_valid_sha256(
    tmp_path: Path, invalid_hash: object
) -> None:
    pool = _pool(tmp_path)
    _approve_higgs(pool, ("spk_001", "spk_003"))
    registry = json.loads(pool["registry"].read_text(encoding="utf-8"))
    registry["speakers"][0]["higgs_reference"]["sha256"] = invalid_hash
    _write(pool["registry"], registry)

    with pytest.raises(
        materialize.SpeakerMaterializationError,
        match="higgs_reference.sha256 must be exactly 64 hexadecimal characters",
    ):
        _run(
            tmp_path,
            pool,
            {"a.json": _dialogue("dialogue_a")},
            [
                {
                    "dialogue_id": "dialogue_a",
                    "role_assignments": {
                        "counsellor": "spk_003",
                        "caller": "spk_001",
                    },
                }
            ],
            higgs_ready=True,
        )


@pytest.mark.parametrize("reference_wav", [None, "", "  ", 7])
def test_higgs_reference_requires_non_empty_wav_path(
    tmp_path: Path, reference_wav: object
) -> None:
    pool = _pool(tmp_path)
    _approve_higgs(pool, ("spk_001", "spk_003"))
    registry = json.loads(pool["registry"].read_text(encoding="utf-8"))
    registry["speakers"][0]["higgs_reference"]["reference_wav"] = reference_wav
    _write(pool["registry"], registry)

    with pytest.raises(
        materialize.SpeakerMaterializationError,
        match="higgs_reference.reference_wav must be a non-empty string",
    ):
        _run(
            tmp_path,
            pool,
            {"a.json": _dialogue("dialogue_a")},
            [
                {
                    "dialogue_id": "dialogue_a",
                    "role_assignments": {
                        "counsellor": "spk_003",
                        "caller": "spk_001",
                    },
                }
            ],
            higgs_ready=True,
        )


def test_higgs_reference_hash_mismatch_fails(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    _approve_higgs(pool, ("spk_001", "spk_003"))
    (tmp_path / "higgs_refs" / "spk_001.wav").write_bytes(b"changed")

    with pytest.raises(
        materialize.SpeakerMaterializationError,
        match="SHA-256 mismatch for spk_001 Higgs reference",
    ):
        _run(
            tmp_path,
            pool,
            {"a.json": _dialogue("dialogue_a")},
            [
                {
                    "dialogue_id": "dialogue_a",
                    "role_assignments": {
                        "counsellor": "spk_003",
                        "caller": "spk_001",
                    },
                }
            ],
            higgs_ready=True,
        )


def test_missing_approved_higgs_reference_file_fails(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    _approve_higgs(pool, ("spk_001", "spk_003"))
    (tmp_path / "higgs_refs" / "spk_001.wav").unlink()

    with pytest.raises(
        materialize.SpeakerMaterializationError,
        match="Approved Higgs reference WAV missing for spk_001",
    ):
        _run(
            tmp_path,
            pool,
            {"a.json": _dialogue("dialogue_a")},
            [
                {
                    "dialogue_id": "dialogue_a",
                    "role_assignments": {
                        "counsellor": "spk_003",
                        "caller": "spk_001",
                    },
                }
            ],
            higgs_ready=True,
        )


def test_same_role_can_map_to_different_speakers_across_dialogues(
    tmp_path: Path,
) -> None:
    pool = _pool(tmp_path)
    dialogues = {
        "a.json": _dialogue("dialogue_a"),
        "b.json": _dialogue("dialogue_b"),
    }
    rows = [
        {
            "dialogue_id": "dialogue_a",
            "role_assignments": {"counsellor": "spk_003", "caller": "spk_001"},
        },
        {
            "dialogue_id": "dialogue_b",
            "role_assignments": {"counsellor": "spk_006", "caller": "spk_009"},
        },
    ]
    _run(tmp_path, pool, dialogues, rows)
    rendered_a = json.loads((tmp_path / "render_input" / "a.json").read_text())
    rendered_b = json.loads((tmp_path / "render_input" / "b.json").read_text())
    assert rendered_a["turns"][1]["speaker"] == "spk_001"
    assert rendered_b["turns"][1]["speaker"] == "spk_009"
    config = yaml.safe_load((tmp_path / "materialized_config.yaml").read_text())
    assert "caller" not in config["tts"]["cosyvoice"]["voice_map"]
    assert "counsellor" not in config["tts"]["cosyvoice"]["voice_map"]


def test_unknown_speaker_fails(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    dialogues = {"a.json": _dialogue("dialogue_a")}
    rows = [
        {
            "dialogue_id": "dialogue_a",
            "role_assignments": {"counsellor": "spk_003", "caller": "spk_999"},
        }
    ]
    with pytest.raises(
        materialize.SpeakerMaterializationError, match="Unknown speaker_id"
    ):
        _run(tmp_path, pool, dialogues, rows)


def test_excluded_speaker_fails(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    dialogues = {"a.json": _dialogue("dialogue_a")}
    rows = [
        {
            "dialogue_id": "dialogue_a",
            "role_assignments": {"counsellor": "spk_003", "caller": EXCLUDED},
        }
    ]
    with pytest.raises(
        materialize.SpeakerMaterializationError, match="excluded from the active pool"
    ):
        _run(tmp_path, pool, dialogues, rows)


def test_missing_role_assignment_fails(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    dialogues = {"a.json": _dialogue("dialogue_a")}
    rows = [
        {
            "dialogue_id": "dialogue_a",
            "role_assignments": {"caller": "spk_001"},
        }
    ]
    with pytest.raises(
        materialize.SpeakerMaterializationError,
        match="missing role assignment for: counsellor",
    ):
        _run(tmp_path, pool, dialogues, rows)


def test_extra_assignment_dialogue_fails(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    dialogues = {"a.json": _dialogue("dialogue_a")}
    rows = [
        {
            "dialogue_id": "dialogue_a",
            "role_assignments": {"counsellor": "spk_003", "caller": "spk_001"},
        },
        {
            "dialogue_id": "dialogue_ghost",
            "role_assignments": {"counsellor": "spk_006", "caller": "spk_009"},
        },
    ]
    with pytest.raises(
        materialize.SpeakerMaterializationError, match="unknown dialogue_id"
    ):
        _run(tmp_path, pool, dialogues, rows)


def test_extra_assignment_role_fails(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    dialogues = {"a.json": _dialogue("dialogue_a")}
    rows = [
        {
            "dialogue_id": "dialogue_a",
            "role_assignments": {
                "counsellor": "spk_003",
                "caller": "spk_001",
                "support_person": "spk_006",
            },
        }
    ]
    with pytest.raises(materialize.SpeakerMaterializationError, match="extra role"):
        _run(tmp_path, pool, dialogues, rows)


def test_missing_reference_wav_fails(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    (tmp_path / "refs" / "spk_001.wav").unlink()
    dialogues = {"a.json": _dialogue("dialogue_a")}
    rows = [
        {
            "dialogue_id": "dialogue_a",
            "role_assignments": {"counsellor": "spk_003", "caller": "spk_001"},
        }
    ]
    with pytest.raises(
        materialize.SpeakerMaterializationError, match="Reference WAV missing"
    ):
        _run(tmp_path, pool, dialogues, rows)


def test_reference_hash_mismatch_fails(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    (tmp_path / "refs" / "spk_001.wav").write_bytes(b"CHANGED")
    dialogues = {"a.json": _dialogue("dialogue_a")}
    rows = [
        {
            "dialogue_id": "dialogue_a",
            "role_assignments": {"counsellor": "spk_003", "caller": "spk_001"},
        }
    ]
    with pytest.raises(
        materialize.SpeakerMaterializationError, match="SHA-256 mismatch"
    ):
        _run(tmp_path, pool, dialogues, rows)


@pytest.mark.parametrize("higgs_ready", [False, True])
def test_materialization_is_deterministic(tmp_path: Path, higgs_ready: bool) -> None:
    pool = _pool(tmp_path)
    if higgs_ready:
        _approve_higgs(pool, ACTIVE)
    dialogues = {
        "b.json": _dialogue("dialogue_b"),
        "a.json": _dialogue("dialogue_a"),
    }
    rows = [
        {
            "dialogue_id": "dialogue_b",
            "role_assignments": {"counsellor": "spk_009", "caller": "spk_006"},
        },
        {
            "dialogue_id": "dialogue_a",
            "role_assignments": {"counsellor": "spk_003", "caller": "spk_001"},
        },
    ]
    first = _run(
        tmp_path,
        pool,
        dialogues,
        rows,
        output_name="out_a",
        higgs_ready=higgs_ready,
    )
    materialize.materialize_speaker_assignments(
        input_path=tmp_path / "canonical",
        assignments_path=tmp_path / "speaker_assignments.jsonl",
        registry_path=pool["registry"],
        active_speakers_path=pool["active"],
        output_dir=tmp_path / "out_b",
        config_out=tmp_path / "config_b.yaml",
        base_config_path=pool["base_config"],
        manifest_path=tmp_path / "manifest_b.json",
        project_root=pool["root"],
        higgs_ready=higgs_ready,
    )
    for name in ("a.json", "b.json"):
        assert (tmp_path / "out_a" / name).read_bytes() == (
            tmp_path / "out_b" / name
        ).read_bytes()
    assert (tmp_path / "materialized_config.yaml").read_bytes() == (
        tmp_path / "config_b.yaml"
    ).read_bytes()
    left = json.loads(Path(first["manifest_path"]).read_text())
    right = json.loads((tmp_path / "manifest_b.json").read_text())
    left["output_dir"] = "x"
    right["output_dir"] = "x"
    left["config_out"] = "x"
    right["config_out"] = "x"
    for row in left["dialogues"] + right["dialogues"]:
        row["materialized_input_path"] = Path(row["materialized_input_path"]).name
    assert left == right


def test_preserves_dialogue_semantics(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    source = _dialogue(
        "dialogue_a",
        caller_text="Exact caller text.",
        counsellor_text="Exact counsellor text.",
        rate="fast",
        arousal="high",
        coarse_affect="anxious",
    )
    _run(
        tmp_path,
        pool,
        {"a.json": source},
        [
            {
                "dialogue_id": "dialogue_a",
                "role_assignments": {"counsellor": "spk_003", "caller": "spk_001"},
            }
        ],
    )
    rendered = json.loads((tmp_path / "render_input" / "a.json").read_text())
    assert rendered["dialogue_id"] == "dialogue_a"
    assert [turn["text"] for turn in rendered["turns"]] == [
        turn["text"] for turn in source["turns"]
    ]
    assert [turn["label"] for turn in rendered["turns"]] == [
        turn["label"] for turn in source["turns"]
    ]
    assert [turn["acoustic_spec"] for turn in rendered["turns"]] == [
        turn["acoustic_spec"] for turn in source["turns"]
    ]
    assert [turn["turn_id"] for turn in rendered["turns"]] == [1, 2]
    assert [turn["speaker"] for turn in rendered["turns"]] == ["spk_003", "spk_001"]


def test_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    stale = tmp_path / "render_input"
    stale.mkdir()
    (stale / "stale.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(materialize.SpeakerMaterializationError, match="not empty"):
        _run(
            tmp_path,
            pool,
            {"a.json": _dialogue("dialogue_a")},
            [
                {
                    "dialogue_id": "dialogue_a",
                    "role_assignments": {
                        "counsellor": "spk_003",
                        "caller": "spk_001",
                    },
                }
            ],
        )


@pytest.mark.parametrize("higgs_ready", [False, True])
def test_cli_wrapper_writes_outputs(tmp_path: Path, higgs_ready: bool) -> None:
    pool = _pool(tmp_path)
    if higgs_ready:
        _approve_higgs(pool, ("spk_001", "spk_003"))
    _write(tmp_path / "canonical" / "a.json", _dialogue("dialogue_a"))
    _assignments(
        tmp_path / "speaker_assignments.jsonl",
        [
            {
                "dialogue_id": "dialogue_a",
                "role_assignments": {"counsellor": "spk_003", "caller": "spk_001"},
            }
        ],
    )
    arguments = [
        "--input",
        str(tmp_path / "canonical"),
        "--assignments",
        str(tmp_path / "speaker_assignments.jsonl"),
        "--registry",
        str(pool["registry"]),
        "--active-speakers",
        str(pool["active"]),
        "--output",
        str(tmp_path / "render_input"),
        "--config-out",
        str(tmp_path / "config.yaml"),
        "--base-config",
        str(pool["base_config"]),
        "--project-root",
        str(pool["root"]),
    ]
    if higgs_ready:
        arguments.append("--higgs-ready")
    exit_code = materialize.main(arguments)
    assert exit_code == 0
    assert (tmp_path / "render_input" / "a.json").is_file()
    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "materialization_manifest.json").is_file()
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert ("higgs" in config["tts"]) is higgs_ready
