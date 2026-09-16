#!/usr/bin/env python3
"""Standard-library JSON-lines worker for an isolated Higgs SGLang server.

The parent process must continuously drain or redirect this worker's stderr,
which also carries the owned SGLang server's stdout and stderr.
"""

from __future__ import annotations

import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any, TextIO

PROTOCOL_STDOUT = sys.stdout
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18080
DEFAULT_STARTUP_TIMEOUT_SECONDS = 900.0
DEFAULT_INFERENCE_TIMEOUT_SECONDS = 300.0
HEALTH_POLL_INTERVAL_SECONDS = 0.25
HEALTH_REQUEST_TIMEOUT_SECONDS = 5.0
TERMINATE_TIMEOUT_SECONDS = 20.0
KILL_TIMEOUT_SECONDS = 20.0

OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}

FROZEN_GENERATION_FIELDS: dict[str, Any] = {
    "voice": "default",
    "response_format": "wav",
    "speed": 1.0,
    "stream": False,
    "max_new_tokens": 2048,
    "temperature": 0.8,
    "top_p": 0.8,
    "top_k": 30,
}

HTTP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class FatalServerError(RuntimeError):
    """Raised when the owned SGLang process can no longer serve requests."""


def _diagnose(message: str) -> None:
    try:
        print(message, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass


def _respond(payload: dict[str, Any]) -> None:
    PROTOCOL_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    PROTOCOL_STDOUT.flush()


def _read_message(stream: TextIO) -> dict[str, Any] | None:
    line = stream.readline()
    if not line:
        return None
    message = json.loads(line)
    if not isinstance(message, dict):
        raise TypeError("worker message must be a JSON object")
    return message


def _nonempty_string(message: dict[str, Any], name: str) -> str:
    value = message.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_number(message: dict[str, Any], name: str, default: float) -> float:
    value = message.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _validate_init(message: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "server_executable",
        "model_dir",
        "host",
        "port",
        "startup_timeout_seconds",
        "inference_timeout_seconds",
    }
    extra = set(message) - allowed
    if extra:
        raise ValueError(f"unsupported initialization fields: {sorted(extra)}")

    server_executable = Path(_nonempty_string(message, "server_executable")).resolve()
    if not server_executable.is_file():
        raise FileNotFoundError(f"server_executable is not a file: {server_executable}")
    if not os.access(server_executable, os.X_OK):
        raise PermissionError(
            f"server_executable is not executable: {server_executable}"
        )

    model_dir = Path(_nonempty_string(message, "model_dir")).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model_dir is not a directory: {model_dir}")

    host = message.get("host", DEFAULT_HOST)
    if host != DEFAULT_HOST:
        raise ValueError(f"host must be the frozen loopback address {DEFAULT_HOST}")
    port = message.get("port", DEFAULT_PORT)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("port must be an integer from 1 to 65535")

    return {
        "server_executable": server_executable,
        "model_dir": model_dir,
        "host": host,
        "port": port,
        "startup_timeout_seconds": _positive_number(
            message,
            "startup_timeout_seconds",
            DEFAULT_STARTUP_TIMEOUT_SECONDS,
        ),
        "inference_timeout_seconds": _positive_number(
            message,
            "inference_timeout_seconds",
            DEFAULT_INFERENCE_TIMEOUT_SECONDS,
        ),
    }


def _validate_request(message: dict[str, Any]) -> dict[str, Any]:
    expected = {"model_input", "reference_wav", "output_path"}
    if set(message) != expected:
        missing = sorted(expected - set(message))
        extra = sorted(set(message) - expected)
        raise ValueError(f"invalid synthesis fields; missing={missing}, extra={extra}")

    model_input = _nonempty_string(message, "model_input")
    reference_wav = Path(_nonempty_string(message, "reference_wav")).resolve()
    if not reference_wav.is_file():
        raise FileNotFoundError(f"reference_wav is not a file: {reference_wav}")
    output_path = Path(_nonempty_string(message, "output_path")).resolve()
    return {
        "model_input": model_input,
        "reference_wav": reference_wav,
        "output_path": output_path,
    }


def _server_environment(server_executable: Path) -> dict[str, str]:
    environment = os.environ.copy()
    existing_path = environment.get("PATH", "")
    environment["PATH"] = str(server_executable.parent)
    if existing_path:
        environment["PATH"] += os.pathsep + existing_path
    environment.update(OFFLINE_ENVIRONMENT)
    return environment


def _launch_server(config: dict[str, Any]) -> subprocess.Popen[bytes]:
    command = [
        str(config["server_executable"]),
        "serve",
        "--model-path",
        str(config["model_dir"]),
        "--host",
        config["host"],
        "--port",
        str(config["port"]),
    ]
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=sys.stderr,
        stderr=sys.stderr,
        env=_server_environment(config["server_executable"]),
        start_new_session=True,
    )


def _check_port_available(host: str, port: int) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, port))
    except OSError as error:
        raise OSError(f"cannot own loopback port {host}:{port}: {error}") from error


def _http_error_detail(error: urllib.error.HTTPError) -> str:
    try:
        try:
            body = error.read().decode("utf-8", errors="replace")
            return f"HTTP {error.code}: {body}"
        except (http.client.IncompleteRead, OSError, ValueError) as body_error:
            return f"HTTP {error.code} (error body unavailable: {body_error})"
    finally:
        try:
            error.close()
        except (OSError, ValueError) as close_error:
            _diagnose(
                f"Higgs worker could not close HTTP error response: {close_error}"
            )


def _wait_for_readiness(
    server: subprocess.Popen[bytes], config: dict[str, Any]
) -> None:
    deadline = time.monotonic() + config["startup_timeout_seconds"]
    health_url = f"http://{config['host']}:{config['port']}/health"
    last_error = "no health response received"
    while True:
        if server.poll() is not None:
            raise FatalServerError(
                f"SGLang server exited before readiness with code {server.returncode}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            with HTTP_OPENER.open(
                health_url,
                timeout=min(HEALTH_REQUEST_TIMEOUT_SECONDS, max(0.01, remaining)),
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if (
                    response.status == 200
                    and payload.get("status") == "healthy"
                    and payload.get("running") is True
                ):
                    return
                last_error = f"health response was not ready: {payload!r}"
        except urllib.error.HTTPError as error:
            last_error = _http_error_detail(error)
        except (
            urllib.error.URLError,
            http.client.IncompleteRead,
            OSError,
            TimeoutError,
            ValueError,
        ) as error:
            last_error = str(error)
        if server.poll() is not None:
            raise FatalServerError(
                f"SGLang server exited before readiness with code {server.returncode}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(HEALTH_POLL_INTERVAL_SECONDS, remaining))
    raise TimeoutError(
        f"SGLang server did not become ready within "
        f"{config['startup_timeout_seconds']} seconds: {last_error}"
    )


def _speech_payload(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "input": request["model_input"],
        **FROZEN_GENERATION_FIELDS,
        "references": [{"audio_path": str(request["reference_wav"])}],
    }


def _wav_metadata(path: Path) -> dict[str, int]:
    try:
        with wave.open(str(path), "rb") as reader:
            sample_rate = reader.getframerate()
            channels = reader.getnchannels()
            samples = reader.getnframes()
    except (wave.Error, EOFError) as error:
        raise ValueError(f"SGLang response is not a readable WAV: {error}") from error
    if sample_rate <= 0 or channels <= 0 or samples <= 0:
        raise ValueError(
            "SGLang WAV metadata must have positive sample rate, channels, and samples"
        )
    return {"sample_rate": sample_rate, "channels": channels, "samples": samples}


def _ensure_server_alive(server: subprocess.Popen[bytes]) -> None:
    if server.poll() is not None:
        raise FatalServerError(
            f"SGLang server exited unexpectedly with code {server.returncode}"
        )


def _synthesise(
    server: subprocess.Popen[bytes], config: dict[str, Any], message: dict[str, Any]
) -> dict[str, Any]:
    _ensure_server_alive(server)
    request = _validate_request(message)
    body = json.dumps(_speech_payload(request), ensure_ascii=False).encode("utf-8")
    http_request = urllib.request.Request(
        f"http://{config['host']}:{config['port']}/v1/audio/speech",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "audio/wav"},
        method="POST",
    )
    try:
        with HTTP_OPENER.open(
            http_request, timeout=config["inference_timeout_seconds"]
        ) as response:
            content_type = response.headers.get_content_type().lower()
            audio = response.read()
            if response.status != 200 or content_type != "audio/wav":
                raise ValueError(
                    "SGLang speech response must be HTTP 200 audio/wav; "
                    f"got {response.status} {content_type!r}"
                )
    except urllib.error.HTTPError as error:
        detail = _http_error_detail(error)
        _ensure_server_alive(server)
        raise RuntimeError(f"SGLang speech request returned {detail}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        _ensure_server_alive(server)
        raise RuntimeError(f"SGLang speech transport failed: {error}") from error

    output_path: Path = request["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".part")
    try:
        temporary.write_bytes(audio)
        metadata = _wav_metadata(temporary)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"status": "ok", "output_path": str(output_path), **metadata}


def _stop_server(server: subprocess.Popen[bytes] | None) -> None:
    if server is None:
        return
    try:
        if server.poll() is not None:
            return
    except OSError as error:
        _diagnose(f"Higgs worker cleanup status check failed: {error}")
    try:
        os.killpg(server.pid, signal.SIGTERM)
    except OSError as error:
        _diagnose(f"Higgs worker cleanup SIGTERM failed: {error}")
    try:
        server.wait(timeout=TERMINATE_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    except OSError as error:
        _diagnose(f"Higgs worker cleanup wait failed: {error}")
    try:
        os.killpg(server.pid, signal.SIGKILL)
    except OSError as error:
        _diagnose(f"Higgs worker cleanup SIGKILL failed: {error}")
    try:
        server.wait(timeout=KILL_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _diagnose("Higgs worker cleanup timed out after SIGKILL; server may remain")
    except OSError as error:
        _diagnose(f"Higgs worker cleanup reap failed: {error}")


def _request_cleanup(signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


def main() -> int:
    signal.signal(signal.SIGINT, _request_cleanup)
    signal.signal(signal.SIGTERM, _request_cleanup)
    server: subprocess.Popen[bytes] | None = None
    try:
        try:
            init = _read_message(sys.stdin)
            if init is None:
                raise RuntimeError("worker received no initialization message")
            config = _validate_init(init)
            _check_port_available(config["host"], config["port"])
            server = _launch_server(config)
            _wait_for_readiness(server, config)
        except Exception as error:  # noqa: BLE001 - report across process boundary
            _respond({"status": "error", "message": f"{type(error).__name__}: {error}"})
            return 1

        _respond({"status": "ready"})
        while True:
            try:
                message = _read_message(sys.stdin)
                if message is None:
                    return 0
                response = _synthesise(server, config, message)
            except FatalServerError as error:
                _respond(
                    {
                        "status": "error",
                        "message": f"{type(error).__name__}: {error}",
                        "fatal": True,
                    }
                )
                return 1
            except Exception as error:  # noqa: BLE001 - keep healthy server reusable
                if server.poll() is not None:
                    _respond(
                        {
                            "status": "error",
                            "message": (
                                "FatalServerError: SGLang server exited unexpectedly "
                                f"with code {server.returncode}; request error was "
                                f"{type(error).__name__}: {error}"
                            ),
                            "fatal": True,
                        }
                    )
                    return 1
                response = {
                    "status": "error",
                    "message": f"{type(error).__name__}: {error}",
                }
            _respond(response)
    except KeyboardInterrupt:
        return 130
    finally:
        _stop_server(server)


if __name__ == "__main__":
    raise SystemExit(main())
