"""Backend identity and metadata descriptions without runtime side effects."""

import hashlib
from pathlib import Path
from typing import Any

from .config import get_engine
from .cosyvoice_controls import (
    COSYVOICE_CONTROL_MAPPING_NAME,
    COSYVOICE_CONTROL_MAPPING_STATUS,
    COSYVOICE_CONTROL_MAPPING_VERSION,
)
from .higgs_controls import (
    HIGGS_CONTROL_CONTRACT_SHA256,
    HIGGS_CONTROL_SOURCE_COMMIT,
    HIGGS_CONTROL_SOURCE_REPOSITORY,
    load_higgs_control_contract,
)

_COSYVOICE3_EXPECTED_SAMPLE_RATE = 24_000
_SAMPLE_RATE_NOT_RUNTIME_VERIFIED = "not_runtime_verified"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _reference_sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _higgs_reference_identities(higgs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    root = _project_root()
    references = {}
    for speaker, voice in higgs["voice_map"].items():
        configured_path = voice["reference_wav"]
        resolved_path = _resolve_path(root, configured_path)
        digest = None
        try:
            if resolved_path.is_file():
                digest = _reference_sha256(resolved_path)
        except OSError:
            pass
        references[speaker] = {
            "reference_wav": configured_path,
            "sha256": digest,
        }
    return references


def _higgs_mapping_identity() -> dict[str, Any]:
    contract = load_higgs_control_contract()
    return {
        "mapping_version": contract["mapping_version"],
        "release_status": contract["release_status"],
        "provenance": {
            "source_repository": HIGGS_CONTROL_SOURCE_REPOSITORY,
            "source_commit": HIGGS_CONTROL_SOURCE_COMMIT,
            "contract_sha256": HIGGS_CONTROL_CONTRACT_SHA256,
        },
    }


def backend_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Return the stable backend identity shared by metadata and resume checks."""
    engine = get_engine(config)
    identity: dict[str, Any] = {"backend": engine}
    if engine == "kokoro":
        identity["model"] = "Kokoro-82M"
    elif engine == "cosyvoice":
        identity.update(
            {
                "model": "Fun-CosyVoice3-0.5B",
                "control_mapping": {
                    "name": COSYVOICE_CONTROL_MAPPING_NAME,
                    "version": COSYVOICE_CONTROL_MAPPING_VERSION,
                    "status": COSYVOICE_CONTROL_MAPPING_STATUS,
                },
            }
        )
    elif engine == "higgs":
        higgs = config["tts"]["higgs"]
        contract = load_higgs_control_contract()
        references = _higgs_reference_identities(higgs)
        identity.update(
            {
                "model_id": contract["model"]["model_id"],
                "model_dir": higgs["model_dir"],
                "server_executable": higgs["server_executable"],
                "ffmpeg_bin": higgs.get("ffmpeg_bin", "ffmpeg"),
                "control_mapping": _higgs_mapping_identity(),
                "references": references,
                "identity_complete": all(
                    reference["sha256"] is not None for reference in references.values()
                ),
            }
        )
    return identity


def describe_engine(config: dict[str, Any]) -> dict[str, Any]:
    """Return an engine and voice configuration snapshot for metadata traceability."""
    engine = get_engine(config)
    identity = backend_identity(config)
    if engine == "kokoro":
        kokoro = config["tts"]["kokoro"]
        return {
            "engine": "kokoro",
            "backend_identity": identity,
            "model": "Kokoro-82M",
            "sample_rate": kokoro["sample_rate"],
            "voices": kokoro["voice_map"],
        }
    if engine == "cosyvoice":
        cosy_cfg = config["tts"]["cosyvoice"]
        configured_sample_rate = cosy_cfg.get("sample_rate")
        return {
            "engine": "cosyvoice",
            "backend_identity": identity,
            "model": "Fun-CosyVoice3-0.5B",
            "mode": "per_turn",
            "available_modes": ["zero_shot", "instruct2"],
            "control_mapping": COSYVOICE_CONTROL_MAPPING_STATUS,
            "model_dir": cosy_cfg.get("model_dir", "models/Fun-CosyVoice3-0.5B"),
            "repo_dir": cosy_cfg.get("repo_dir", "third_party/CosyVoice"),
            "fp16": cosy_cfg.get("fp16", True),
            "load_trt": cosy_cfg.get("load_trt", False),
            "load_vllm": cosy_cfg.get("load_vllm", False),
            "expected_sample_rate": (
                configured_sample_rate
                if configured_sample_rate is not None
                else _COSYVOICE3_EXPECTED_SAMPLE_RATE
            ),
            "expected_sample_rate_source": (
                "config" if configured_sample_rate is not None else "model_default"
            ),
            "runtime_sample_rate": None,
            "sample_rate_verification": _SAMPLE_RATE_NOT_RUNTIME_VERIFIED,
            "voices": {
                speaker: voice["prompt_wav"]
                for speaker, voice in cosy_cfg["voice_map"].items()
            },
            "prompts": {
                speaker: {
                    "prompt_wav": voice["prompt_wav"],
                    "prompt_text": voice["prompt_text"],
                }
                for speaker, voice in cosy_cfg["voice_map"].items()
            },
        }
    if engine == "higgs":
        higgs = config["tts"]["higgs"]
        return {
            "engine": "higgs",
            "backend_identity": identity,
            "mode": "per_turn",
            "model_id": identity["model_id"],
            "model_dir": higgs["model_dir"],
            "control_mapping": _higgs_mapping_identity(),
            "runtime": {
                "server_executable": higgs["server_executable"],
                "host": higgs.get("host", "127.0.0.1"),
                "port": higgs.get("port", 18080),
                "startup_timeout_seconds": higgs.get("startup_timeout_seconds", 900),
                "inference_timeout_seconds": higgs.get(
                    "inference_timeout_seconds", 300
                ),
                "ffmpeg_bin": higgs.get("ffmpeg_bin", "ffmpeg"),
            },
            "runtime_verification": "not_runtime_verified",
            "references": identity["references"],
        }

    raise ValueError(f"Engine description is not implemented for {engine!r}")
