"""Final-schema CosyVoice3 preflight and cached-plan execution."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..render_models import CosyVoicePreparedTurn
from . import cosyvoice_client as cosyvoice


def _runtime(config: dict[str, Any]) -> tuple[dict[str, Any], Path, Path, Path]:
    try:
        tts = config["tts"]
        settings = tts["cosyvoice"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "Prepared CosyVoice execution requires tts.cosyvoice"
        ) from error
    if tts.get("engine") != "cosyvoice" or not isinstance(settings, dict):
        raise RuntimeError(
            "Prepared CosyVoice execution requires tts.engine='cosyvoice'"
        )
    root = cosyvoice._project_root()
    repo = cosyvoice._resolve_path(
        root, settings.get("repo_dir", "third_party/CosyVoice")
    )
    model = cosyvoice._resolve_path(
        root, settings.get("model_dir", "models/Fun-CosyVoice3-0.5B")
    )
    python = cosyvoice._resolve_path(
        root,
        settings.get("python_bin", "third_party/CosyVoice/.venv/bin/python"),
    )
    missing = [
        f"{label}: {path}"
        for label, path in (
            ("CosyVoice worker", cosyvoice._WORKER_SCRIPT),
            ("CosyVoice Python interpreter", python),
            ("CosyVoice repository", repo),
            ("CosyVoice model directory", model),
        )
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(
            "CosyVoice cannot start because required paths are missing: "
            + "; ".join(missing)
        )
    return settings, python, repo, model


def preflight_prepared_turn(
    turn: CosyVoicePreparedTurn, config: dict[str, Any]
) -> tuple[dict[str, Any], Path, Path, Path]:
    """Verify capabilities and the selected reference without starting a worker."""
    plan = turn.plan
    pause_within = plan["normalized"]["pause_within_count"]
    if pause_within:
        raise RuntimeError(
            f"Prepared CosyVoice turn {turn.ordinal} requests pause_within="
            f"{pause_within}, but no evidence-backed deterministic CosyVoice3 "
            "realization is available"
        )
    reference = turn.resolved_prompt_wav
    if not reference.is_file():
        raise RuntimeError(
            f"Prepared CosyVoice prompt audio is missing or not a file: {reference}"
        )
    with reference.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != turn.reference_sha256:
        raise RuntimeError(f"Prepared CosyVoice prompt SHA-256 mismatch: {reference}")
    expected = turn.worker_request(Path("output.wav"))
    if (
        expected["text"] != plan["cosyvoice"]["text"]
        or expected["speed"] != plan["cosyvoice"]["speed"]
        or expected["mode"] != plan["cosyvoice"]["inference_mode"]
    ):
        raise RuntimeError(f"Prepared CosyVoice turn {turn.ordinal} request is invalid")
    return _runtime(config)


def synthesize_prepared_turn(
    turn: CosyVoicePreparedTurn, out_dir: Path, config: dict[str, Any]
) -> Path:
    """Send only the cached plan and selected reference to the existing worker."""
    settings, python, repo, model = preflight_prepared_turn(turn, config)
    output = out_dir / f"turn_{turn.ordinal:03d}.wav"
    worker = cosyvoice._get_worker(
        str(python),
        str(repo),
        str(model),
        settings.get("load_trt", False),
        settings.get("load_vllm", False),
        settings.get("fp16", True),
    )
    response = cosyvoice._request(worker, turn.worker_request(output.resolve()))
    if response.get("status") != "ok":
        raise RuntimeError(
            f"CosyVoice failed for prepared turn {turn.ordinal}: "
            f"{response.get('message')}"
        )
    returned = response.get("output_path")
    if not isinstance(returned, str) or Path(returned).resolve() != output.resolve():
        raise RuntimeError(
            f"CosyVoice returned unexpected output_path for turn {turn.ordinal}: "
            f"{returned}"
        )
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError(f"CosyVoice did not produce a non-empty WAV: {output}")
    return output
