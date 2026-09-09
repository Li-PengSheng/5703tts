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


def _run(
    tmp_path: Path,
    pool: dict[str, Path],
    dialogues: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    output_name: str = "render_input",
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

    load_and_validate(tmp_path / "render_input" / "a.json", config)
    load_and_validate(tmp_path / "render_input" / "b.json", config)
    assert manifest["input_count"] == 2
    assert "placeholder" not in voice_map


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


def test_materialization_is_deterministic(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
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
    first = _run(tmp_path, pool, dialogues, rows, output_name="out_a")
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


def test_cli_wrapper_writes_outputs(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
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
    exit_code = materialize.main(
        [
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
    )
    assert exit_code == 0
    assert (tmp_path / "render_input" / "a.json").is_file()
    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "materialization_manifest.json").is_file()
