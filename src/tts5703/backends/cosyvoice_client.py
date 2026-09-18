"""Reusable parent-side client for the isolated CosyVoice worker protocol."""

from __future__ import annotations

import atexit
import json
import logging
import subprocess
import threading
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger("tts5703.tts_engine")

# ponytail: isolated from the legacy backend so that path can be deleted next phase;
# consolidate only if legacy unexpectedly remains after the migration.
_WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "cosyvoice_worker.py"
_TERMINATE_TIMEOUT_SEC = 5


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _drain_stderr(proc: subprocess.Popen[str]) -> deque[str]:
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
    """Start once and reuse the CosyVoice worker for this process."""
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
    proc.stdin.write(
        json.dumps(
            {
                "repo_dir": repo_dir,
                "model_dir": model_dir,
                "load_trt": load_trt,
                "load_vllm": load_vllm,
                "fp16": fp16,
            }
        )
        + "\n"
    )
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
