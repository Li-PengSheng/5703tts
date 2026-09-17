"""Parent-side Higgs worker lifecycle, request validation, and rate processing."""

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
from pathlib import Path
from typing import Any

from ..higgs_controls import resolve_higgs_turn_controls
from ..render_plan import PreparedTurn
from ..validate import NormalizedTurn

logger = logging.getLogger("tts5703.tts_engine")

_WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "higgs_worker.py"
# The accepted worker can spend up to 20 seconds on SIGTERM and another 20 on
# SIGKILL while reaping its SGLang process group. Give stdin EOF and SIGTERM
# enough time to complete that ownership contract before escalating.
_GRACEFUL_SHUTDOWN_TIMEOUT_SEC = 45
_TERMINATE_TIMEOUT_SEC = 45
_KILL_TIMEOUT_SEC = 5
_STDERR_TAIL_LINES = 100
_STDERR_FRAGMENT_CHARS = 2048


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _drain_stderr(proc: subprocess.Popen[str]) -> deque[str]:
    """Continuously drain worker/server diagnostics into a bounded tail."""
    tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)

    def record(fragment: str) -> None:
        message = fragment.rstrip("\r\n")
        if message:
            tail.append(message)
            logger.debug("event=higgs_worker_stderr message=%s", message)

    def drain() -> None:
        assert proc.stderr is not None
        try:
            while fragment := proc.stderr.readline(_STDERR_FRAGMENT_CHARS + 1):
                record(fragment)
        except (OSError, UnicodeError, ValueError) as error:
            logger.warning("event=higgs_worker_stderr_read_failed error=%s", error)
            raw_stderr = getattr(proc.stderr, "buffer", None)
            if raw_stderr is None:
                return
            try:
                while chunk := raw_stderr.read(_STDERR_FRAGMENT_CHARS + 1):
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


def _wait_for_stderr(proc: subprocess.Popen[str]) -> None:
    thread = getattr(proc, "_higgs_stderr_thread", None)
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=1)


def _worker_error(proc: subprocess.Popen[str], fallback: str) -> str:
    if proc.poll() is not None:
        _wait_for_stderr(proc)
    tail = getattr(proc, "_higgs_stderr_tail", deque())
    diagnostics = "\n".join(list(tail))
    return f"{fallback}\n{diagnostics}" if diagnostics else fallback


def _terminate_worker(proc: subprocess.Popen[str]) -> None:
    """Give the worker time to clean SGLang, then terminate and reap it."""
    if proc.poll() is not None:
        proc.wait()
        _wait_for_stderr(proc)
        return
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except OSError:
        pass
    try:
        proc.wait(timeout=_GRACEFUL_SHUTDOWN_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        try:
            proc.terminate()
            proc.wait(timeout=_TERMINATE_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=_KILL_TIMEOUT_SEC)
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
    _wait_for_stderr(proc)


class _StartupError(RuntimeError):
    """Carry a startup protocol diagnostic through the cleanup boundary."""


_WorkerKey = tuple[str, str, str, int, float, float]
_worker_lock = threading.RLock()
_worker_cache: tuple[_WorkerKey, subprocess.Popen[str]] | None = None


def _shutdown_worker() -> None:
    global _worker_cache
    try:
        with _worker_lock:
            cached = _worker_cache
            if cached is None:
                return
            _terminate_worker(cached[1])
            _worker_cache = None
    except Exception:  # noqa: BLE001, S110 - best-effort interpreter cleanup
        pass


def _discard_worker(proc: subprocess.Popen[str]) -> None:
    global _worker_cache
    with _worker_lock:
        _terminate_worker(proc)
        if _worker_cache is not None and _worker_cache[1] is proc:
            _worker_cache = None


def _get_worker(
    server_executable: str,
    model_dir: str,
    host: str,
    port: int,
    startup_timeout_seconds: float,
    inference_timeout_seconds: float,
) -> subprocess.Popen[str]:
    """Start and exclusively cache one worker that owns one SGLang server."""
    global _worker_cache
    key: _WorkerKey = (
        server_executable,
        model_dir,
        host,
        port,
        startup_timeout_seconds,
        inference_timeout_seconds,
    )
    with _worker_lock:
        if _worker_cache is not None:
            cached_key, cached_proc = _worker_cache
            if cached_key == key:
                return cached_proc
            _terminate_worker(cached_proc)
            _worker_cache = None

        proc: subprocess.Popen[str] | None = None
        try:
            proc = subprocess.Popen(
                [sys.executable, str(_WORKER_SCRIPT)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            proc._higgs_stderr_tail = _drain_stderr(proc)
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
                raise _StartupError(
                    f"invalid startup data: {response_line.rstrip()!r}"
                ) from error
            if not isinstance(response, dict) or response.get("status") != "ready":
                fallback = (
                    str(response.get("message"))
                    if isinstance(response, dict) and response.get("message")
                    else f"worker exited with code {proc.poll()}"
                )
                raise _StartupError(fallback)
            logger.debug("event=higgs_worker_ready model_dir=%s", model_dir)
        except BaseException as error:
            if proc is not None:
                _terminate_worker(proc)
            if isinstance(error, _StartupError):
                assert proc is not None
                message = _worker_error(proc, str(error))
                raise RuntimeError(  # noqa: TRY004 - subprocess protocol failure
                    f"Failed to start Higgs worker: {message}"
                ) from error
            raise
        _worker_cache = (key, proc)
        return proc


atexit.register(_shutdown_worker)


def _request(proc: subprocess.Popen[str], request: dict[str, Any]) -> dict[str, Any]:
    """Send one request; retain the worker only after recoverable errors."""
    if proc.poll() is not None:
        fallback = f"worker exited with code {proc.returncode}"
        _discard_worker(proc)
        raise RuntimeError(
            f"Higgs worker exited unexpectedly: {_worker_error(proc, fallback)}"
        )
    assert proc.stdin is not None and proc.stdout is not None
    try:
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
    except (BrokenPipeError, OSError, ValueError) as error:
        _discard_worker(proc)
        message = _worker_error(proc, str(error))
        raise RuntimeError(f"Higgs worker request channel failed: {message}") from error

    response_line = proc.stdout.readline()
    if not response_line:
        fallback = f"worker exited with code {proc.poll()}"
        _discard_worker(proc)
        raise RuntimeError(
            f"Higgs worker closed its output: {_worker_error(proc, fallback)}"
        )
    try:
        response = json.loads(response_line)
    except json.JSONDecodeError as error:
        fallback = f"invalid response data: {response_line.rstrip()!r}"
        _discard_worker(proc)
        message = _worker_error(proc, fallback)
        raise RuntimeError(f"Higgs worker protocol failed: {message}") from error
    if not isinstance(response, dict):
        fallback = f"non-object response: {response!r}"
        _discard_worker(proc)
        message = _worker_error(proc, fallback)
        raise RuntimeError(  # noqa: TRY004 - protocol boundary, not caller type use
            f"Higgs worker protocol failed: {message}"
        )
    if response.get("status") == "error":
        message = str(response.get("message") or "unknown worker error")
        if response.get("fatal") is True:
            _discard_worker(proc)
            message = _worker_error(proc, message)
            raise RuntimeError(f"Higgs worker fatal error: {message}")
        raise RuntimeError(f"Higgs synthesis request failed: {message}")
    if response.get("status") != "ok":
        fallback = f"unexpected response: {response!r}"
        _discard_worker(proc)
        message = _worker_error(proc, fallback)
        raise RuntimeError(f"Higgs worker protocol failed: {message}")
    return response


def _validate_wav(path: Path) -> None:
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


def _validate_success(
    proc: subprocess.Popen[str], response: dict[str, Any], expected_raw_path: Path
) -> None:
    returned = response.get("output_path")
    if not isinstance(returned, str) or not returned.strip():
        message = "successful response has no output_path"
    elif Path(returned).resolve() != expected_raw_path.resolve():
        message = f"worker returned unexpected output_path: {returned}"
    else:
        try:
            _validate_wav(expected_raw_path)
            return
        except RuntimeError as error:
            message = str(error)
    _discard_worker(proc)
    message = _worker_error(proc, message)
    raise RuntimeError(f"Higgs worker protocol failed: {message}")


def _resolve_command(root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(root / path) if path.parent != Path(".") else value


def _apply_rate(
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
        "-c:a",
        "pcm_s16le",
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
        _validate_wav(processed)
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


def synthesize_turn(
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
    if not _WORKER_SCRIPT.is_file():
        missing.append(f"Higgs worker: {_WORKER_SCRIPT}")
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
    worker = _get_worker(
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
        response = _request(worker, request)
        _validate_success(worker, response, raw_path)
    except Exception:
        raw_path.unlink(missing_ok=True)
        processed_path.unlink(missing_ok=True)
        raise

    _apply_rate(
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


def _prepared_runtime(
    config: dict[str, Any],
) -> tuple[dict[str, Any], Path, Path, str]:
    try:
        tts = config["tts"]
        higgs = tts["higgs"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("Prepared Higgs execution requires tts.higgs") from error
    if tts.get("engine") != "higgs":
        raise RuntimeError("Prepared Controlled TTS v1 execution requires Higgs")
    if not isinstance(higgs, dict):
        raise RuntimeError(  # noqa: TRY004 - execution boundary failure
            "Prepared Higgs execution requires tts.higgs mapping"
        )
    root = _project_root()
    for field in ("server_executable", "model_dir"):
        value = higgs.get(field)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"Prepared Higgs execution requires tts.higgs.{field}")
    host = higgs.get("host", "127.0.0.1")
    port = higgs.get("port", 18080)
    if host != "127.0.0.1":
        raise RuntimeError("tts.higgs.host must be 127.0.0.1")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise RuntimeError("tts.higgs.port must be an integer from 1 to 65535")
    for field, default in (
        ("startup_timeout_seconds", 900),
        ("inference_timeout_seconds", 300),
    ):
        value = higgs.get(field, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise RuntimeError(f"tts.higgs.{field} must be a positive number")
    server_executable = _resolve_path(root, higgs["server_executable"]).resolve()
    model_dir = _resolve_path(root, higgs["model_dir"]).resolve()
    missing = []
    if not _WORKER_SCRIPT.is_file():
        missing.append(f"Higgs worker: {_WORKER_SCRIPT}")
    if not server_executable.is_file() or not os.access(server_executable, os.X_OK):
        missing.append(f"Higgs server executable: {server_executable}")
    if not model_dir.is_dir():
        missing.append(f"Higgs model directory: {model_dir}")
    if missing:
        raise RuntimeError(
            "Higgs cannot start because required runtime paths are missing: "
            + "; ".join(missing)
        )
    ffmpeg_bin = higgs.get("ffmpeg_bin", "ffmpeg")
    if not isinstance(ffmpeg_bin, str) or not ffmpeg_bin.strip():
        raise RuntimeError("tts.higgs.ffmpeg_bin must be a nonblank string")
    return higgs, server_executable, model_dir, _resolve_command(root, ffmpeg_bin)


def preflight_prepared_turn(
    turn: PreparedTurn, config: dict[str, Any]
) -> tuple[dict[str, Any], Path, Path, str]:
    """Fail closed on prepared execution inputs without starting the worker."""
    runtime = _prepared_runtime(config)
    if not turn.model_input.strip():
        raise RuntimeError(f"Prepared turn {turn.ordinal} has empty model_input")
    if turn.synthesis_call_count != 1:
        raise RuntimeError(
            f"Prepared turn {turn.ordinal} synthesis_call_count must equal 1"
        )
    if turn.rate not in {"slow", "normal", "fast"}:
        raise RuntimeError(f"Prepared turn {turn.ordinal} has unsupported rate")
    expected_factor = {"slow": 0.85, "normal": None, "fast": 1.15}[turn.rate]
    rate_plan = turn.rate_plan
    if rate_plan.get("factor") != expected_factor or rate_plan.get("enabled") is not (
        expected_factor is not None
    ):
        raise RuntimeError(f"Prepared turn {turn.ordinal} has invalid rate plan")
    reference = turn.resolved_reference_wav
    if not reference.is_file():
        raise RuntimeError(
            f"Prepared Higgs reference audio is missing or not a file: {reference}"
        )
    with reference.open("rb") as handle:
        actual_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual_hash != turn.reference_sha256:
        raise RuntimeError(f"Prepared Higgs reference SHA-256 mismatch: {reference}")
    return runtime


def synthesize_prepared_turn(
    turn: PreparedTurn, out_dir: Path, config: dict[str, Any]
) -> Path:
    """Execute one cached final plan with one unchanged worker request."""
    higgs, server_executable, model_dir, ffmpeg_bin = preflight_prepared_turn(
        turn, config
    )
    final_path = out_dir / f"turn_{turn.ordinal:03d}.wav"
    raw_path = out_dir / f"turn_{turn.ordinal:03d}.higgs_raw.wav"
    processed_path = final_path.with_name(f"{final_path.stem}.higgs_processed.part.wav")
    raw_path.unlink(missing_ok=True)
    processed_path.unlink(missing_ok=True)
    worker = _get_worker(
        str(server_executable),
        str(model_dir),
        higgs.get("host", "127.0.0.1"),
        higgs.get("port", 18080),
        higgs.get("startup_timeout_seconds", 900),
        higgs.get("inference_timeout_seconds", 300),
    )
    request = {
        "model_input": turn.model_input,
        "reference_wav": str(turn.resolved_reference_wav),
        "output_path": str(raw_path.resolve()),
    }
    logger.debug(
        "event=prepared_turn_tts_start engine=higgs ordinal=%d speaker=%s rate=%s",
        turn.ordinal,
        turn.render_speaker_id,
        turn.rate,
    )
    try:
        response = _request(worker, request)
        _validate_success(worker, response, raw_path)
    except Exception:
        raw_path.unlink(missing_ok=True)
        processed_path.unlink(missing_ok=True)
        raise
    _apply_rate(raw_path, final_path, turn.rate_plan, ffmpeg_bin)
    logger.debug(
        "event=prepared_turn_tts_complete engine=higgs ordinal=%d output=%s bytes=%d",
        turn.ordinal,
        final_path.name,
        final_path.stat().st_size,
    )
    return final_path
