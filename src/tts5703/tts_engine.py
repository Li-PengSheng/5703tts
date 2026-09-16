"""Stage 2 TTS adapter, selectable through ``tts.engine``."""

import atexit
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import wave
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .config import VALID_ENGINES
from .cosyvoice_controls import (
    COSYVOICE_CONTROL_MAPPING_NAME,
    COSYVOICE_CONTROL_MAPPING_STATUS,
    COSYVOICE_CONTROL_MAPPING_VERSION,
    BackendControlError,
    resolve_cosyvoice_controls,
    validate_cosyvoice_controls,
)
from .higgs_controls import (
    HIGGS_CONTROL_CONTRACT_SHA256,
    HIGGS_CONTROL_SOURCE_COMMIT,
    HIGGS_CONTROL_SOURCE_REPOSITORY,
    HiggsControlError,
    load_higgs_control_contract,
    resolve_higgs_controls,
)
from .validate import NormalizedTurn

logger = logging.getLogger(__name__)


def get_engine(config: dict[str, Any]) -> str:
    engine = config["tts"].get("engine", "edge_tts")
    if engine not in VALID_ENGINES:
        raise ValueError(
            f"Unsupported tts.engine: {engine} (available: {sorted(VALID_ENGINES)})"
        )
    return engine


def _project_root() -> Path:
    """Repository root, two levels above this file (src/tts5703/tts_engine.py)."""
    return Path(__file__).resolve().parents[2]


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def turn_audio_extension(config: dict[str, Any]) -> str:
    """Return the per-turn container extension emitted by the selected engine."""
    return ".mp3" if get_engine(config) == "edge_tts" else ".wav"


_EDGE_TTS_SEMANTIC_RATES = {
    "slow": "-20%",
    "normal": "+0%",
    "fast": "+20%",
}
_KOKORO_SEMANTIC_RATES = {
    "slow": 0.8,
    "normal": 1.0,
    "fast": 1.2,
}


def rate_to_edge_tts(rate: str) -> str:
    """Map semantic rates to EdgeTTS percentages, preserving legacy values."""
    return _EDGE_TTS_SEMANTIC_RATES.get(rate, rate)


def rate_to_kokoro_speed(rate: str) -> float:
    """Map semantic or legacy percentage rates to a Kokoro speed multiplier."""
    if rate in _KOKORO_SEMANTIC_RATES:
        return _KOKORO_SEMANTIC_RATES[rate]
    return max(0.1, 1 + int(rate[:-1]) / 100)


def _rate_to_kokoro_speed(rate: str) -> float:
    """Backward-compatible alias for the pre-v0.2 internal helper."""
    return rate_to_kokoro_speed(rate)


def preflight_cosyvoice_controls(turn: NormalizedTurn) -> None:
    """Reject requested controls CosyVoice cannot map, before synthesis starts."""
    validate_cosyvoice_controls(turn.arousal, turn.coarse_affect)


def resolve_higgs_turn_controls(turn: Any) -> dict[str, Any]:
    """Resolve the deterministic Higgs plan used by synthesis and metadata."""
    return resolve_higgs_controls(
        text=turn.text,
        rate=turn.rate,
        arousal=turn.arousal,
        coarse_affect=turn.coarse_affect,
        pause_before_ms=turn.pause_before_ms,
        pause_after_ms=turn.pause_after_ms,
        speaker_id=turn.speaker,
    )


def preflight_higgs_controls(turn: NormalizedTurn) -> None:
    """Resolve one Higgs control plan without starting a model or worker."""
    try:
        resolve_higgs_turn_controls(turn)
    except HiggsControlError as error:
        raise BackendControlError(str(error)) from error


def preflight_backend_controls(turn: NormalizedTurn, config: dict[str, Any]) -> None:
    """Validate one turn against the selected backend's control mappings.

    Backends that ignore a control (see ``engine_capabilities``) must not reject
    it here; the requested value is preserved and reported as ignored instead.
    """
    engine = get_engine(config)
    if engine == "cosyvoice":
        preflight_cosyvoice_controls(turn)
    elif engine == "higgs":
        preflight_higgs_controls(turn)


def preflight_dialogue_controls(
    turns: list[NormalizedTurn], config: dict[str, Any]
) -> None:
    """Fail a whole dialogue before spending any synthesis time on it."""
    for turn in turns:
        try:
            preflight_backend_controls(turn, config)
        except BackendControlError as error:
            raise BackendControlError(f"turn {turn.turn_id}: {error}") from error


def build_cosyvoice_request(
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


@lru_cache(maxsize=4)
def _get_kokoro_pipeline(lang_code: str, device: str | None):
    # Lazy import keeps EdgeTTS runs independent of Kokoro model loading.
    from kokoro import KPipeline

    return KPipeline(lang_code=lang_code, device=device)


_COSYVOICE_WORKER_SCRIPT = Path(__file__).resolve().parent / "cosyvoice_worker.py"
_COSYVOICE_TERMINATE_TIMEOUT_SEC = 5
_HIGGS_WORKER_SCRIPT = Path(__file__).resolve().parent / "higgs_worker.py"
# The accepted worker can spend up to 20 seconds on SIGTERM and another 20 on
# SIGKILL while reaping its SGLang process group. Give stdin EOF and SIGTERM
# enough time to complete that ownership contract before escalating.
_HIGGS_GRACEFUL_SHUTDOWN_TIMEOUT_SEC = 45
_HIGGS_TERMINATE_TIMEOUT_SEC = 45
_HIGGS_KILL_TIMEOUT_SEC = 5
_HIGGS_STDERR_TAIL_LINES = 100
_HIGGS_STDERR_FRAGMENT_CHARS = 2048
# Official output rate of Fun-CosyVoice3-0.5B. This is the expected rate, not an
# observed one: the worker reports its own rate at startup, but that value is not
# threaded back through synthesize_turn.
_COSYVOICE3_EXPECTED_SAMPLE_RATE = 24_000
_SAMPLE_RATE_NOT_RUNTIME_VERIFIED = "not_runtime_verified"


def _drain_cosyvoice_stderr(proc: subprocess.Popen[str]) -> deque[str]:
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


def _cosyvoice_worker_error(
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
def _get_cosyvoice_worker(
    python_bin: str,
    repo_dir: str,
    model_dir: str,
    load_trt: bool,
    load_vllm: bool,
    fp16: bool,
) -> subprocess.Popen[str]:
    """Start (once) and reuse the CosyVoice worker subprocess for this run.

    CosyVoice3's dependencies live in their own virtual environment,
    separate from this project's .venv (see ``tts.cosyvoice.python_bin``),
    so synthesis happens in a subprocess speaking JSON-lines over stdio
    rather than an in-process import.
    """
    proc = subprocess.Popen(
        [python_bin, str(_COSYVOICE_WORKER_SCRIPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stderr_tail = _drain_cosyvoice_stderr(proc)
    proc._cosyvoice_stderr_tail = stderr_tail
    atexit.register(_shutdown_cosyvoice_worker, proc)

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
        message = response.get("message") or _cosyvoice_worker_error(
            proc, stderr_tail, f"worker exited with code {proc.poll()}"
        )
        if proc.poll() is None:
            proc.kill()
        raise RuntimeError(
            f"Failed to start CosyVoice worker (python_bin={python_bin}): {message}"
        )
    logger.debug("event=cosyvoice_worker_ready model_dir=%s", model_dir)
    return proc


def _shutdown_cosyvoice_worker(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.terminate()
        except Exception:  # noqa: BLE001, S110 - best-effort interpreter cleanup
            pass


def _terminate_cosyvoice_worker(proc: subprocess.Popen[str]) -> None:
    """Terminate and reap a worker whose protocol stream is no longer usable.

    Without this, a worker that stopped answering but is still alive keeps its
    GPU memory for the rest of the batch, because the cached handle is dropped
    and nothing else ever waits on the process.
    """
    if proc.poll() is not None:
        return
    try:
        if proc.stdin:
            proc.stdin.close()
    except OSError:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=_COSYVOICE_TERMINATE_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            # The worker exited between the timeout and the kill; it still needs
            # reaping below, so this race is not a cleanup failure.
            logger.debug(
                "event=cosyvoice_worker_already_gone pid=%s",
                getattr(proc, "pid", None),
            )
        try:
            proc.wait(timeout=_COSYVOICE_TERMINATE_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            logger.warning(
                "event=cosyvoice_worker_unreaped pid=%s", getattr(proc, "pid", None)
            )
    except OSError:
        pass


def _cosyvoice_request(
    proc: subprocess.Popen[str], request: dict[str, Any]
) -> dict[str, Any]:
    """Send one JSON-line request and synchronously read its matching response.

    The production pipeline is sequential, so the protocol needs neither request
    identifiers nor concurrent-response routing. EOF invalidates and reaps the
    cached worker; a worker-reported synthesis error is returned to the caller and
    leaves the process available for a later dialogue.
    """
    if proc.poll() is not None:
        stderr_tail = getattr(proc, "_cosyvoice_stderr_tail", deque())
        message = _cosyvoice_worker_error(
            proc, stderr_tail, f"worker exited with code {proc.returncode}"
        )
        _get_cosyvoice_worker.cache_clear()
        raise RuntimeError(f"CosyVoice worker exited unexpectedly: {message}")
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()
    response_line = proc.stdout.readline()
    if not response_line:
        stderr_tail = getattr(proc, "_cosyvoice_stderr_tail", deque())
        message = _cosyvoice_worker_error(
            proc, stderr_tail, f"worker exited with code {proc.poll()}"
        )
        # Reap before dropping the cached handle so the next call starts a fresh
        # worker instead of competing with a hung one for GPU memory.
        _terminate_cosyvoice_worker(proc)
        _get_cosyvoice_worker.cache_clear()
        raise RuntimeError(f"CosyVoice worker closed its output: {message}")
    try:
        return json.loads(response_line)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "CosyVoice worker returned invalid response data: "
            f"{response_line.rstrip()!r}"
        ) from error


def _drain_higgs_stderr(proc: subprocess.Popen[str]) -> deque[str]:
    """Continuously drain worker/server diagnostics into a bounded tail."""
    tail: deque[str] = deque(maxlen=_HIGGS_STDERR_TAIL_LINES)

    def record(fragment: str) -> None:
        message = fragment.rstrip("\r\n")
        if message:
            tail.append(message)
            logger.debug("event=higgs_worker_stderr message=%s", message)

    def drain() -> None:
        assert proc.stderr is not None
        try:
            while fragment := proc.stderr.readline(_HIGGS_STDERR_FRAGMENT_CHARS + 1):
                record(fragment)
        except (OSError, UnicodeError, ValueError) as error:
            logger.warning("event=higgs_worker_stderr_read_failed error=%s", error)
            raw_stderr = getattr(proc.stderr, "buffer", None)
            if raw_stderr is None:
                return
            try:
                while chunk := raw_stderr.read(_HIGGS_STDERR_FRAGMENT_CHARS + 1):
                    record(chunk.decode("utf-8", errors="replace"))
            except (OSError, ValueError) as fallback_error:
                logger.warning(
                    "event=higgs_worker_stderr_fallback_failed error=%s",
                    fallback_error,
                )

    thread = threading.Thread(target=drain, name="higgs-stderr", daemon=True)
    proc._higgs_stderr_thread = thread
    thread.start()
    return tail


def _wait_for_higgs_stderr(proc: subprocess.Popen[str]) -> None:
    thread = getattr(proc, "_higgs_stderr_thread", None)
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=1)


def _higgs_worker_error(proc: subprocess.Popen[str], fallback: str) -> str:
    if proc.poll() is not None:
        _wait_for_higgs_stderr(proc)
    tail = getattr(proc, "_higgs_stderr_tail", deque())
    diagnostics = "\n".join(list(tail))
    return f"{fallback}\n{diagnostics}" if diagnostics else fallback


def _terminate_higgs_worker(proc: subprocess.Popen[str]) -> None:
    """Give the worker time to clean SGLang, then terminate and reap it."""
    if proc.poll() is not None:
        proc.wait()
        _wait_for_higgs_stderr(proc)
        return
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except OSError:
        pass
    try:
        proc.wait(timeout=_HIGGS_GRACEFUL_SHUTDOWN_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        try:
            proc.terminate()
            proc.wait(timeout=_HIGGS_TERMINATE_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=_HIGGS_KILL_TIMEOUT_SEC)
            except (OSError, subprocess.TimeoutExpired):
                logger.warning(
                    "event=higgs_worker_unreaped pid=%s", getattr(proc, "pid", None)
                )
        except OSError:
            if proc.poll() is None:
                logger.warning(
                    "event=higgs_worker_terminate_failed pid=%s",
                    getattr(proc, "pid", None),
                )
    _wait_for_higgs_stderr(proc)


class _HiggsStartupError(RuntimeError):
    """Carry a startup protocol diagnostic through the cleanup boundary."""


_HiggsWorkerKey = tuple[str, str, str, int, float, float]
_higgs_worker_lock = threading.RLock()
_higgs_worker_cache: tuple[_HiggsWorkerKey, subprocess.Popen[str]] | None = None


def _shutdown_higgs_worker() -> None:
    global _higgs_worker_cache
    try:
        with _higgs_worker_lock:
            cached = _higgs_worker_cache
            if cached is None:
                return
            _terminate_higgs_worker(cached[1])
            _higgs_worker_cache = None
    except Exception:  # noqa: BLE001, S110 - best-effort interpreter cleanup
        pass


def _discard_higgs_worker(proc: subprocess.Popen[str]) -> None:
    global _higgs_worker_cache
    with _higgs_worker_lock:
        _terminate_higgs_worker(proc)
        if _higgs_worker_cache is not None and _higgs_worker_cache[1] is proc:
            _higgs_worker_cache = None


def _get_higgs_worker(
    server_executable: str,
    model_dir: str,
    host: str,
    port: int,
    startup_timeout_seconds: float,
    inference_timeout_seconds: float,
) -> subprocess.Popen[str]:
    """Start and exclusively cache one worker that owns one SGLang server."""
    global _higgs_worker_cache
    key: _HiggsWorkerKey = (
        server_executable,
        model_dir,
        host,
        port,
        startup_timeout_seconds,
        inference_timeout_seconds,
    )
    with _higgs_worker_lock:
        if _higgs_worker_cache is not None:
            cached_key, cached_proc = _higgs_worker_cache
            if cached_key == key:
                return cached_proc
            _terminate_higgs_worker(cached_proc)
            _higgs_worker_cache = None

        proc: subprocess.Popen[str] | None = None
        try:
            proc = subprocess.Popen(
                [sys.executable, str(_HIGGS_WORKER_SCRIPT)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            proc._higgs_stderr_tail = _drain_higgs_stderr(proc)
            assert proc.stdin is not None and proc.stdout is not None
            init = {
                "server_executable": server_executable,
                "model_dir": model_dir,
                "host": host,
                "port": port,
                "startup_timeout_seconds": startup_timeout_seconds,
                "inference_timeout_seconds": inference_timeout_seconds,
            }
            proc.stdin.write(json.dumps(init) + "\n")
            proc.stdin.flush()
            response_line = proc.stdout.readline()
            try:
                response = json.loads(response_line) if response_line else None
            except json.JSONDecodeError as error:
                raise _HiggsStartupError(
                    f"invalid startup data: {response_line.rstrip()!r}"
                ) from error
            if not isinstance(response, dict) or response.get("status") != "ready":
                fallback = (
                    str(response.get("message"))
                    if isinstance(response, dict) and response.get("message")
                    else f"worker exited with code {proc.poll()}"
                )
                raise _HiggsStartupError(fallback)
            logger.debug("event=higgs_worker_ready model_dir=%s", model_dir)
        except BaseException as error:
            if proc is not None:
                _terminate_higgs_worker(proc)
            if isinstance(error, _HiggsStartupError):
                assert proc is not None
                message = _higgs_worker_error(proc, str(error))
                raise RuntimeError(  # noqa: TRY004 - subprocess protocol failure
                    f"Failed to start Higgs worker: {message}"
                ) from error
            raise
        _higgs_worker_cache = (key, proc)
        return proc


atexit.register(_shutdown_higgs_worker)


def _higgs_request(
    proc: subprocess.Popen[str], request: dict[str, Any]
) -> dict[str, Any]:
    """Send one request; retain the worker only after recoverable errors."""
    if proc.poll() is not None:
        fallback = f"worker exited with code {proc.returncode}"
        _discard_higgs_worker(proc)
        raise RuntimeError(
            f"Higgs worker exited unexpectedly: {_higgs_worker_error(proc, fallback)}"
        )
    assert proc.stdin is not None and proc.stdout is not None
    try:
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
    except (BrokenPipeError, OSError, ValueError) as error:
        _discard_higgs_worker(proc)
        message = _higgs_worker_error(proc, str(error))
        raise RuntimeError(f"Higgs worker request channel failed: {message}") from error

    response_line = proc.stdout.readline()
    if not response_line:
        fallback = f"worker exited with code {proc.poll()}"
        _discard_higgs_worker(proc)
        raise RuntimeError(
            f"Higgs worker closed its output: {_higgs_worker_error(proc, fallback)}"
        )
    try:
        response = json.loads(response_line)
    except json.JSONDecodeError as error:
        fallback = f"invalid response data: {response_line.rstrip()!r}"
        _discard_higgs_worker(proc)
        message = _higgs_worker_error(proc, fallback)
        raise RuntimeError(f"Higgs worker protocol failed: {message}") from error
    if not isinstance(response, dict):
        fallback = f"non-object response: {response!r}"
        _discard_higgs_worker(proc)
        message = _higgs_worker_error(proc, fallback)
        raise RuntimeError(  # noqa: TRY004 - protocol boundary, not caller type use
            f"Higgs worker protocol failed: {message}"
        )
    if response.get("status") == "error":
        message = str(response.get("message") or "unknown worker error")
        if response.get("fatal") is True:
            _discard_higgs_worker(proc)
            message = _higgs_worker_error(proc, message)
            raise RuntimeError(f"Higgs worker fatal error: {message}")
        raise RuntimeError(f"Higgs synthesis request failed: {message}")
    if response.get("status") != "ok":
        fallback = f"unexpected response: {response!r}"
        _discard_higgs_worker(proc)
        message = _higgs_worker_error(proc, fallback)
        raise RuntimeError(f"Higgs worker protocol failed: {message}")
    return response


def _validate_higgs_wav(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Higgs worker did not produce a non-empty WAV: {path}")
    try:
        with wave.open(str(path), "rb") as reader:
            sample_rate = reader.getframerate()
            channels = reader.getnchannels()
            samples = reader.getnframes()
    except (EOFError, wave.Error) as error:
        raise RuntimeError(
            f"Higgs worker output is not a readable WAV: {path}"
        ) from error
    if sample_rate <= 0 or channels <= 0 or samples <= 0:
        raise RuntimeError(f"Higgs worker output WAV has invalid metadata: {path}")


def _validate_higgs_success(
    proc: subprocess.Popen[str], response: dict[str, Any], expected_raw_path: Path
) -> None:
    returned = response.get("output_path")
    if not isinstance(returned, str) or not returned.strip():
        message = "successful response has no output_path"
    elif Path(returned).resolve() != expected_raw_path.resolve():
        message = f"worker returned unexpected output_path: {returned}"
    else:
        try:
            _validate_higgs_wav(expected_raw_path)
            return
        except RuntimeError as error:
            message = str(error)
    _discard_higgs_worker(proc)
    message = _higgs_worker_error(proc, message)
    raise RuntimeError(f"Higgs worker protocol failed: {message}")


def _resolve_command(root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(root / path) if path.parent != Path(".") else value


def _apply_higgs_rate(
    raw_path: Path,
    final_path: Path,
    rate_plan: dict[str, Any],
    ffmpeg_bin: str,
) -> None:
    """Publish raw audio atomically, optionally through frozen FFmpeg atempo."""
    if not rate_plan["enabled"]:
        raw_path.replace(final_path)
        return

    factor = rate_plan["factor"]
    processed = final_path.with_name(f"{final_path.stem}.higgs_processed.part.wav")
    processed.unlink(missing_ok=True)
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(raw_path),
        "-af",
        f"atempo={factor}",
        str(processed),
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Higgs rate postprocess failed with code {result.returncode}: "
                f"{result.stderr.strip()}"
            )
        _validate_higgs_wav(processed)
        processed.replace(final_path)
        raw_path.unlink(missing_ok=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Higgs FFmpeg executable was not found: {ffmpeg_bin}"
        ) from error
    except OSError as error:
        raise RuntimeError(f"Higgs FFmpeg could not be started: {error}") from error
    finally:
        processed.unlink(missing_ok=True)


def _synthesize_higgs_turn(
    turn: NormalizedTurn, out_dir: Path, config: dict[str, Any]
) -> Path:
    higgs = config["tts"]["higgs"]
    root = _project_root()
    reference_wav = _resolve_path(
        root, higgs["voice_map"][turn.speaker]["reference_wav"]
    ).resolve()
    if not reference_wav.is_file():
        raise RuntimeError(
            f"Higgs reference audio is missing or not a file: {reference_wav}"
        )

    server_executable = _resolve_path(root, higgs["server_executable"]).resolve()
    model_dir = _resolve_path(root, higgs["model_dir"]).resolve()
    missing = []
    if not _HIGGS_WORKER_SCRIPT.is_file():
        missing.append(f"Higgs worker: {_HIGGS_WORKER_SCRIPT}")
    if not server_executable.is_file() or not os.access(server_executable, os.X_OK):
        missing.append(f"Higgs server executable: {server_executable}")
    if not model_dir.is_dir():
        missing.append(f"Higgs model directory: {model_dir}")
    if missing:
        raise RuntimeError(
            "Higgs cannot start because required runtime paths are missing: "
            + "; ".join(missing)
        )

    resolution = resolve_higgs_turn_controls(turn)
    final_path = out_dir / f"turn_{turn.turn_id:03d}.wav"
    raw_path = out_dir / f"turn_{turn.turn_id:03d}.higgs_raw.wav"
    processed_path = final_path.with_name(f"{final_path.stem}.higgs_processed.part.wav")
    raw_path.unlink(missing_ok=True)
    processed_path.unlink(missing_ok=True)
    worker = _get_higgs_worker(
        str(server_executable),
        str(model_dir),
        higgs.get("host", "127.0.0.1"),
        higgs.get("port", 18080),
        higgs.get("startup_timeout_seconds", 900),
        higgs.get("inference_timeout_seconds", 300),
    )
    request = {
        "model_input": resolution["higgs"]["model_input"],
        "reference_wav": str(reference_wav),
        "output_path": str(raw_path.resolve()),
    }
    logger.debug(
        "event=turn_tts_start engine=higgs turn=%d speaker=%s rate=%s",
        turn.turn_id,
        turn.speaker,
        turn.rate,
    )
    try:
        response = _higgs_request(worker, request)
        _validate_higgs_success(worker, response, raw_path)
    except Exception:
        raw_path.unlink(missing_ok=True)
        processed_path.unlink(missing_ok=True)
        raise

    _apply_higgs_rate(
        raw_path,
        final_path,
        resolution["postprocess"]["rate"],
        _resolve_command(root, higgs.get("ffmpeg_bin", "ffmpeg")),
    )
    logger.debug(
        "event=turn_tts_complete engine=higgs turn=%d output=%s bytes=%d",
        turn.turn_id,
        final_path.name,
        final_path.stat().st_size,
    )
    return final_path


@lru_cache(maxsize=2)
def _get_chatterbox_turbo(device: str, model_dir: str | None, nano: bool):
    """Load Turbo/Nano lazily so EdgeTTS/Kokoro runs do not import its model stack."""
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    if model_dir:
        return ChatterboxTurboTTS.from_local(model_dir, device=device)
    return ChatterboxTurboTTS.from_pretrained(device=device, nano=nano)


async def synthesize_turn(
    turn: NormalizedTurn, out_dir: Path, config: dict[str, Any]
) -> Path:
    """Synthesize one turn with the engine selected in ``tts.engine``."""
    engine = get_engine(config)
    if engine == "higgs":
        return _synthesize_higgs_turn(turn, out_dir, config)
    if engine == "edge_tts":
        import edge_tts  # Lazy import: only needed when Edge TTS is selected.

        voice = config["speaker_voice_map"][turn.speaker]
        output_path = out_dir / f"turn_{turn.turn_id:03d}.mp3"
        logger.debug(
            "event=turn_tts_start engine=edge_tts turn=%d speaker=%s voice=%s rate=%s",
            turn.turn_id,
            turn.speaker,
            voice,
            turn.rate,
        )
        await edge_tts.Communicate(
            text=turn.text,
            voice=voice,
            rate=rate_to_edge_tts(turn.rate),
        ).save(str(output_path))
        logger.debug(
            "event=turn_tts_complete engine=edge_tts turn=%d output=%s bytes=%d",
            turn.turn_id,
            output_path.name,
            output_path.stat().st_size,
        )
        return output_path

    output_path = out_dir / f"turn_{turn.turn_id:03d}.wav"
    if engine == "kokoro":
        kokoro = config["tts"]["kokoro"]
        voice = kokoro["voice_map"][turn.speaker]
        logger.debug(
            "event=turn_tts_start engine=kokoro turn=%d speaker=%s voice=%s rate=%s",
            turn.turn_id,
            turn.speaker,
            voice,
            turn.rate,
        )
        pipeline = _get_kokoro_pipeline(kokoro["lang_code"], kokoro.get("device"))
        chunks = [
            np.asarray(result.audio)
            for result in pipeline(
                turn.text,
                voice=voice,
                speed=rate_to_kokoro_speed(turn.rate),
            )
        ]
        if not chunks:
            raise RuntimeError(f"Kokoro did not generate audio for turn {turn.turn_id}")
        sf.write(output_path, np.concatenate(chunks), kokoro["sample_rate"])
        logger.debug(
            "event=turn_tts_complete engine=kokoro turn=%d output=%s chunks=%d bytes=%d",
            turn.turn_id,
            output_path.name,
            len(chunks),
            output_path.stat().st_size,
        )
        return output_path

    if engine == "cosyvoice":
        preflight_cosyvoice_controls(turn)
        cosy_cfg = config["tts"]["cosyvoice"]
        voice = cosy_cfg["voice_map"][turn.speaker]
        root = _project_root()
        repo_dir = _resolve_path(
            root, cosy_cfg.get("repo_dir", "third_party/CosyVoice")
        )
        model_dir = _resolve_path(
            root, cosy_cfg.get("model_dir", "models/Fun-CosyVoice3-0.5B")
        )
        prompt_wav = _resolve_path(root, voice["prompt_wav"])
        python_bin = _resolve_path(
            root, cosy_cfg.get("python_bin", "third_party/CosyVoice/.venv/bin/python")
        )
        required_paths = {
            "CosyVoice worker": _COSYVOICE_WORKER_SCRIPT,
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
        request = build_cosyvoice_request(
            turn,
            voice["prompt_text"],
            prompt_wav,
            output_path,
        )
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
        worker = _get_cosyvoice_worker(
            str(python_bin),
            str(repo_dir),
            str(model_dir),
            cosy_cfg.get("load_trt", False),
            cosy_cfg.get("load_vllm", False),
            cosy_cfg.get("fp16", True),
        )
        response = _cosyvoice_request(worker, request)
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

    turbo = config["tts"]["chatterbox_turbo"]
    reference_audio = turbo.get("reference_audio_map", {}).get(turn.speaker)
    logger.debug(
        "event=turn_tts_start engine=chatterbox_turbo turn=%d speaker=%s reference_audio=%s",
        turn.turn_id,
        turn.speaker,
        reference_audio or "builtin",
    )
    model = _get_chatterbox_turbo(
        turbo["device"], turbo.get("model_dir"), turbo.get("variant", "turbo") == "nano"
    )
    kwargs = {
        "temperature": turbo["temperature"],
        "top_p": turbo["top_p"],
        "top_k": turbo["top_k"],
        "repetition_penalty": turbo["repetition_penalty"],
    }
    if reference_audio:
        kwargs["audio_prompt_path"] = reference_audio
    audio = model.generate(turn.text, **kwargs)
    sf.write(output_path, audio.squeeze(0).detach().cpu().numpy(), model.sr)
    logger.debug(
        "event=turn_tts_complete engine=chatterbox_turbo turn=%d output=%s bytes=%d",
        turn.turn_id,
        output_path.name,
        output_path.stat().st_size,
    )
    return output_path


async def synthesize_all_turns(
    turns: list[NormalizedTurn], out_dir: Path, config: dict[str, Any]
) -> dict[int, Path]:
    """Render exactly one file per turn, sequentially in validated turn order.

    Sequential rendering avoids concurrent use of stateful model instances and
    the single CosyVoice stdio channel. A failure stops this dialogue immediately;
    already-written turn files are intentionally left for diagnosis.
    """
    return {
        turn.turn_id: await synthesize_turn(turn, out_dir, config) for turn in turns
    }


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
    elif engine == "chatterbox_turbo":
        variant = config["tts"]["chatterbox_turbo"].get("variant", "turbo")
        identity["model"] = f"chatterbox-{variant}"
    return identity


def describe_engine(config: dict[str, Any]) -> dict[str, Any]:
    """Return an engine and voice configuration snapshot for metadata traceability."""
    engine = get_engine(config)
    identity = backend_identity(config)
    if engine == "edge_tts":
        return {
            "engine": "edge_tts",
            "backend_identity": identity,
            "voices": config["speaker_voice_map"],
        }
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
            # The worker reports its own sample rate over the protocol, but
            # synthesize_turn returns only a path, so the runtime value cannot be
            # recorded without changing that API. Until then the expected rate is
            # a declaration, never an observation of the written audio.
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

    turbo = config["tts"]["chatterbox_turbo"]
    variant = turbo.get("variant", "turbo")
    return {
        "engine": "chatterbox_turbo",
        "backend_identity": identity,
        "model": f"chatterbox-{variant}",
        "device": turbo["device"],
        "reference_audio": turbo.get("reference_audio_map", {}),
        "generation_params": {
            "temperature": turbo["temperature"],
            "top_p": turbo["top_p"],
            "top_k": turbo["top_k"],
            "repetition_penalty": turbo["repetition_penalty"],
        },
    }
