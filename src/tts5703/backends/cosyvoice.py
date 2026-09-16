"""Parent-side CosyVoice worker lifecycle and turn synthesis."""

import atexit
import json
import logging
import subprocess
import threading
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..cosyvoice_controls import resolve_cosyvoice_controls, validate_cosyvoice_controls
from ..validate import NormalizedTurn

logger = logging.getLogger("tts5703.tts_engine")

_WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "cosyvoice_worker.py"
_TERMINATE_TIMEOUT_SEC = 5


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def preflight_controls(turn: NormalizedTurn) -> None:
    """Reject requested controls CosyVoice cannot map, before synthesis starts."""
    validate_cosyvoice_controls(turn.arousal, turn.coarse_affect)


def build_request(
    turn: NormalizedTurn,
    prompt_text: str,
    prompt_wav: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Translate corpus controls into the isolated CosyVoice worker protocol."""
    resolution = resolve_cosyvoice_controls(turn.rate, turn.arousal, turn.coarse_affect)
    instruction = resolution["resolved_instruction"]
    request: dict[str, Any] = {
        "text": turn.text,
        "prompt_text": prompt_text,
        "prompt_wav": str(prompt_wav),
        "output_path": str(output_path),
        "speed": resolution["speaking_rate"]["speed"],
        "mode": resolution["inference_mode"],
    }
    if instruction is not None:
        request["instruction"] = instruction
    return request


def _drain_stderr(proc: subprocess.Popen[str]) -> deque[str]:
    """Drain worker diagnostics so its stderr pipe cannot block a long batch."""
    tail: deque[str] = deque(maxlen=50)

    def drain() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            message = line.rstrip()
            if message:
                tail.append(message)
                logger.debug("event=cosyvoice_worker_stderr message=%s", message)

    threading.Thread(
        target=drain,
        name="cosyvoice-stderr",
        daemon=True,
    ).start()
    return tail


def _worker_error(
    proc: subprocess.Popen[str], stderr_tail: deque[str], fallback: str
) -> str:
    """Return the most useful available diagnostic from a failed worker."""
    if proc.poll() is None:
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    return "\n".join(stderr_tail) or fallback


@lru_cache(maxsize=1)
def _get_worker(
    python_bin: str,
    repo_dir: str,
    model_dir: str,
    load_trt: bool,
    load_vllm: bool,
    fp16: bool,
) -> subprocess.Popen[str]:
    """Start (once) and reuse the CosyVoice worker subprocess for this run."""
    proc = subprocess.Popen(
        [python_bin, str(_WORKER_SCRIPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stderr_tail = _drain_stderr(proc)
    proc._cosyvoice_stderr_tail = stderr_tail
    atexit.register(_shutdown_worker, proc)

    assert proc.stdin is not None and proc.stdout is not None
    init = {
        "repo_dir": repo_dir,
        "model_dir": model_dir,
        "load_trt": load_trt,
        "load_vllm": load_vllm,
        "fp16": fp16,
    }
    logger.debug(
        "event=cosyvoice_worker_start python_bin=%s model_dir=%s", python_bin, model_dir
    )
    proc.stdin.write(json.dumps(init) + "\n")
    proc.stdin.flush()
    response_line = proc.stdout.readline()
    try:
        response = json.loads(response_line) if response_line else {}
    except json.JSONDecodeError as error:
        proc.kill()
        raise RuntimeError(
            "CosyVoice worker returned invalid startup data: "
            f"{response_line.rstrip()!r}"
        ) from error
    if response.get("status") != "ready":
        message = response.get("message") or _worker_error(
            proc, stderr_tail, f"worker exited with code {proc.poll()}"
        )
        if proc.poll() is None:
            proc.kill()
        raise RuntimeError(
            f"Failed to start CosyVoice worker (python_bin={python_bin}): {message}"
        )
    logger.debug("event=cosyvoice_worker_ready model_dir=%s", model_dir)
    return proc


def _shutdown_worker(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.terminate()
        except Exception:  # noqa: BLE001, S110 - best-effort interpreter cleanup
            pass


def _terminate_worker(proc: subprocess.Popen[str]) -> None:
    """Terminate and reap a worker whose protocol stream is no longer usable."""
    if proc.poll() is not None:
        return
    try:
        if proc.stdin:
            proc.stdin.close()
    except OSError:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=_TERMINATE_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            logger.debug(
                "event=cosyvoice_worker_already_gone pid=%s",
                getattr(proc, "pid", None),
            )
        try:
            proc.wait(timeout=_TERMINATE_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            logger.warning(
                "event=cosyvoice_worker_unreaped pid=%s", getattr(proc, "pid", None)
            )
    except OSError:
        pass


def _request(proc: subprocess.Popen[str], request: dict[str, Any]) -> dict[str, Any]:
    """Send one JSON-line request and synchronously read its matching response."""
    if proc.poll() is not None:
        stderr_tail = getattr(proc, "_cosyvoice_stderr_tail", deque())
        message = _worker_error(
            proc, stderr_tail, f"worker exited with code {proc.returncode}"
        )
        _get_worker.cache_clear()
        raise RuntimeError(f"CosyVoice worker exited unexpectedly: {message}")
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()
    response_line = proc.stdout.readline()
    if not response_line:
        stderr_tail = getattr(proc, "_cosyvoice_stderr_tail", deque())
        message = _worker_error(
            proc, stderr_tail, f"worker exited with code {proc.poll()}"
        )
        _terminate_worker(proc)
        _get_worker.cache_clear()
        raise RuntimeError(f"CosyVoice worker closed its output: {message}")
    try:
        return json.loads(response_line)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "CosyVoice worker returned invalid response data: "
            f"{response_line.rstrip()!r}"
        ) from error


def synthesize_turn(
    turn: NormalizedTurn, out_dir: Path, config: dict[str, Any]
) -> Path:
    preflight_controls(turn)
    cosy_cfg = config["tts"]["cosyvoice"]
    voice = cosy_cfg["voice_map"][turn.speaker]
    root = _project_root()
    repo_dir = _resolve_path(root, cosy_cfg.get("repo_dir", "third_party/CosyVoice"))
    model_dir = _resolve_path(
        root, cosy_cfg.get("model_dir", "models/Fun-CosyVoice3-0.5B")
    )
    prompt_wav = _resolve_path(root, voice["prompt_wav"])
    python_bin = _resolve_path(
        root, cosy_cfg.get("python_bin", "third_party/CosyVoice/.venv/bin/python")
    )
    required_paths = {
        "CosyVoice worker": _WORKER_SCRIPT,
        "CosyVoice Python interpreter": python_bin,
        "CosyVoice repository": repo_dir,
        "CosyVoice model directory": model_dir,
        "CosyVoice prompt audio": prompt_wav,
    }
    missing_paths = [
        f"{label}: {path}"
        for label, path in required_paths.items()
        if not path.exists()
    ]
    if missing_paths:
        raise RuntimeError(
            "CosyVoice cannot start because required paths are missing: "
            + "; ".join(missing_paths)
        )
    output_path = out_dir / f"turn_{turn.turn_id:03d}.wav"
    request = build_request(turn, voice["prompt_text"], prompt_wav, output_path)
    logger.debug(
        "event=turn_tts_start engine=cosyvoice turn=%d speaker=%s "
        "requested_rate=%s speed=%.3f mode=%s has_instruction=%s",
        turn.turn_id,
        turn.speaker,
        turn.rate,
        request["speed"],
        request["mode"],
        "instruction" in request,
    )
    worker = _get_worker(
        str(python_bin),
        str(repo_dir),
        str(model_dir),
        cosy_cfg.get("load_trt", False),
        cosy_cfg.get("load_vllm", False),
        cosy_cfg.get("fp16", True),
    )
    response = _request(worker, request)
    if response.get("status") != "ok":
        raise RuntimeError(
            f"CosyVoice failed for turn {turn.turn_id}: {response.get('message')}"
        )
    logger.debug(
        "event=turn_tts_complete engine=cosyvoice turn=%d output=%s bytes=%d",
        turn.turn_id,
        output_path.name,
        output_path.stat().st_size,
    )
    return output_path
