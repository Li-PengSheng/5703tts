"""Offline lifecycle tests for the isolated Higgs worker and a fake HTTP server."""

from __future__ import annotations

import ast
import io
import json
import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self

import pytest

from tts5703.backends import higgs as higgs_backend
from tts5703.backends import higgs_worker

WORKER = Path("src/tts5703/backends/higgs_worker.py").resolve()
FROZEN_HIGGS_MAPPING = Path(
    "comp5703-tts-experiments/testing/week5_candidate_evaluation/"
    "targeted_revision_v1/mappings/higgs.py"
).resolve()


def test_parent_resolves_relocated_worker() -> None:
    assert higgs_backend._WORKER_SCRIPT == WORKER
    assert WORKER.is_file()


FAKE_SERVER = r"""
import argparse
import io
import json
import os
import signal
import socket
import sys
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("--model-path", required=True)
parser.add_argument("--host", required=True)
parser.add_argument("--port", required=True, type=int)
parser.add_argument("--max-total-tokens", required=True, type=int)
parser.add_argument("--max-running-requests", required=True, type=int)
parser.add_argument("--cuda-graph-max-bs", required=True, type=int)
args = parser.parse_args()
model_dir = Path(args.model_path)
behavior = json.loads((model_dir / "behavior.json").read_text(encoding="utf-8"))
pid_path = model_dir / "server.pid"
request_path = model_dir / "requests.jsonl"
count_path = model_dir / "launch-count.txt"
term_path = model_dir / "terminated.txt"
count = int(count_path.read_text() or "0") if count_path.exists() else 0
count_path.write_text(str(count + 1), encoding="utf-8")
pid_path.write_text(str(os.getpid()), encoding="utf-8")

if behavior.get("read_stdin"):
    (model_dir / "server-stdin.txt").write_text(sys.stdin.read(), encoding="utf-8")

def stop(signum, frame):
    term_path.write_text(str(signum), encoding="utf-8")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
if behavior.get("log_bytes"):
    text = "server-log-" + "x" * int(behavior["log_bytes"])
    print(text, file=sys.stdout, flush=True)
    print(text, file=sys.stderr, flush=True)
if behavior.get("exit_before_ready"):
    raise SystemExit(7)

ready_at = time.monotonic() + float(behavior.get("startup_delay", 0))
request_count = 0
health_count = 0

def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        channels = int(behavior.get("channels", 1))
        frames = int(behavior.get("frames", 240))
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(int(behavior.get("sample_rate", 24000)))
        writer.writeframes(b"\x00\x00" * frames * channels)
    return output.getvalue()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *values):
        print(format % values, file=sys.stderr, flush=True)

    def do_GET(self):
        global health_count
        if self.path != "/health":
            self.send_error(404)
            return
        health_count += 1
        (model_dir / "health-count.txt").write_text(str(health_count), encoding="utf-8")
        sequence = behavior.get("health_sequence", [])
        mode = sequence[health_count - 1] if health_count <= len(sequence) else None
        if health_count <= int(behavior.get("health_http_errors", 0)):
            mode = "http_503"
        if mode == "reset":
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
            return
        if mode == "malformed":
            body = b"{malformed-json"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if mode == "incomplete":
            body = b'{"status":"healthy"'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body) + 100))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            self.close_connection = True
            return
        ready = not behavior.get("never_ready") and time.monotonic() >= ready_at
        payload = {"status": "healthy" if ready else "starting", "running": ready}
        body = json.dumps(payload).encode()
        self.send_response(503 if mode == "http_503" else (200 if ready else 503))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        global request_count
        if self.path != "/v1/audio/speech":
            self.send_error(404)
            return
        request_count += 1
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        record = {
            "payload": payload,
            "content_type": self.headers.get("Content-Type"),
            "accept": self.headers.get("Accept"),
            "pid": os.getpid(),
        }
        with request_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if behavior.get("exit_on_request"):
            os._exit(9)
        if (
            (behavior.get("fail_first") and request_count == 1)
            or request_count <= int(behavior.get("fail_requests", 0))
            or request_count in behavior.get("fail_on_requests", [])
        ):
            body = behavior.get("fail_body", "synthetic request failure").encode()
            self.send_response(int(behavior.get("fail_status", 500)))
            self.send_header("Content-Type", "text/plain")
        elif behavior.get("malformed_first") and request_count == 1:
            body = b"not-a-wav"
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
        else:
            body = wav_bytes()
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/octet-stream" if behavior.get("wrong_content_type") else "audio/wav",
            )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

server = ThreadingHTTPServer((args.host, args.port), Handler)
server.serve_forever()
"""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind((higgs_worker.DEFAULT_HOST, 0))
        return int(probe.getsockname()[1])


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 160)


def _init_paths(tmp_path: Path) -> tuple[Path, Path]:
    executable = tmp_path / "fake-sgl-omni"
    executable.write_text(f"#!{sys.executable}\n{FAKE_SERVER}", encoding="utf-8")
    executable.chmod(0o755)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    return executable, model_dir


def _valid_init(tmp_path: Path) -> dict[str, Any]:
    executable, model_dir = _init_paths(tmp_path)
    return {
        "server_executable": str(executable),
        "model_dir": str(model_dir),
        "host": "127.0.0.1",
        "port": _free_port(),
        "startup_timeout_seconds": 2,
        "inference_timeout_seconds": 2,
    }


class WorkerHarness:
    def __init__(self, tmp_path: Path, behavior: dict[str, Any] | None = None):
        self.root = tmp_path
        self.init = _valid_init(tmp_path)
        self.model_dir = Path(self.init["model_dir"])
        (self.model_dir / "behavior.json").write_text(
            json.dumps(behavior or {}), encoding="utf-8"
        )
        self.reference = tmp_path / "reference.wav"
        _write_wav(self.reference)
        self.stderr: list[str] = []
        self.process = subprocess.Popen(
            [sys.executable, str(WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.process.stderr is not None
        self.stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self.stderr_thread.start()

    def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        self.stderr.extend(self.process.stderr)

    def send(self, message: dict[str, Any]) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def response(self, timeout: float = 5) -> dict[str, Any]:
        assert self.process.stdout is not None
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            raise AssertionError(
                f"worker produced no response; returncode={self.process.poll()} "
                f"stderr={''.join(self.stderr)[-2000:]}"
            )
        line = self.process.stdout.readline()
        if not line:
            raise AssertionError(
                f"worker stdout closed; returncode={self.process.poll()} "
                f"stderr={''.join(self.stderr)[-2000:]}"
            )
        return json.loads(line)

    def start(self, timeout: float = 5) -> dict[str, Any]:
        self.send(self.init)
        return self.response(timeout)

    def request(
        self, output: Path, model_input: str = "Resolved model input"
    ) -> dict[str, Any]:
        self.send(
            {
                "model_input": model_input,
                "reference_wav": str(self.reference),
                "output_path": str(output),
            }
        )
        return self.response()

    def server_pid(self) -> int:
        return int((self.model_dir / "server.pid").read_text(encoding="utf-8"))

    def requests(self) -> list[dict[str, Any]]:
        path = self.model_dir / "requests.jsonl"
        if not path.exists():
            return []
        return [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]

    def close_stdin(self) -> None:
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()

    def cleanup(self) -> None:
        try:
            if self.process.poll() is None:
                self.close_stdin()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=3)
        finally:
            self._cleanup_fake_server()
            self.stderr_thread.join(timeout=1)

    def _cleanup_fake_server(self) -> None:
        pid_path = self.model_dir / "server.pid"
        if not pid_path.exists():
            return
        pid = int(pid_path.read_text(encoding="utf-8"))
        if not _process_exists(pid):
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and _process_exists(pid):
            time.sleep(0.02)
        if _process_exists(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and _process_exists(pid):
                time.sleep(0.02)
        assert not _process_exists(pid), f"fake server process {pid} leaked"


@pytest.fixture
def harnesses() -> list[WorkerHarness]:
    active: list[WorkerHarness] = []
    yield active
    for harness in active:
        harness.cleanup()


def _harness(
    harnesses: list[WorkerHarness],
    tmp_path: Path,
    behavior: dict[str, Any] | None = None,
) -> WorkerHarness:
    harness = WorkerHarness(tmp_path, behavior)
    harnesses.append(harness)
    return harness


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_stopped(pid: int, timeout: float = 3) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and _process_exists(pid):
        time.sleep(0.02)
    assert not _process_exists(pid)


def test_init_defaults_and_frozen_server_environment(tmp_path: Path) -> None:
    message = _valid_init(tmp_path)
    for field in (
        "host",
        "port",
        "startup_timeout_seconds",
        "inference_timeout_seconds",
    ):
        del message[field]

    validated = higgs_worker._validate_init(message)
    environment = higgs_worker._server_environment(validated["server_executable"])

    assert validated["host"] == "127.0.0.1"
    assert validated["port"] == 18080
    assert validated["startup_timeout_seconds"] == 900
    assert validated["inference_timeout_seconds"] == 300
    assert environment["PATH"].split(os.pathsep)[0] == str(
        validated["server_executable"].parent
    )
    assert {key: environment[key] for key in higgs_worker.OFFLINE_ENVIRONMENT} == (
        higgs_worker.OFFLINE_ENVIRONMENT
    )


def test_server_launch_uses_frozen_production_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = higgs_worker._validate_init(_valid_init(tmp_path))
    launched: list[list[str]] = []

    def fake_popen(command: list[str], **kwargs: Any) -> object:
        launched.append(command)
        return object()

    monkeypatch.setattr(higgs_worker.subprocess, "Popen", fake_popen)
    higgs_worker._launch_server(config)

    assert launched == [
        [
            str(config["server_executable"]),
            "serve",
            "--model-path",
            str(config["model_dir"]),
            "--host",
            config["host"],
            "--port",
            str(config["port"]),
            "--max-total-tokens",
            "71680",
            "--max-running-requests",
            "1",
            "--cuda-graph-max-bs",
            "1",
        ]
    ]


def test_init_rejects_missing_and_non_executable_server(tmp_path: Path) -> None:
    message = _valid_init(tmp_path)
    message["server_executable"] = str(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="server_executable"):
        higgs_worker._validate_init(message)

    path = tmp_path / "not-executable"
    path.write_text("x", encoding="utf-8")
    message["server_executable"] = str(path)
    with pytest.raises(PermissionError, match="not executable"):
        higgs_worker._validate_init(message)


def test_init_rejects_missing_model_directory(tmp_path: Path) -> None:
    message = _valid_init(tmp_path)
    message["model_dir"] = str(tmp_path / "missing-model")
    with pytest.raises(FileNotFoundError, match="model_dir"):
        higgs_worker._validate_init(message)


@pytest.mark.parametrize("host", ["localhost", "::1", "0.0.0.0", "example.com"])
def test_init_rejects_any_non_frozen_host(tmp_path: Path, host: str) -> None:
    message = _valid_init(tmp_path)
    message["host"] = host
    with pytest.raises(ValueError, match="127.0.0.1"):
        higgs_worker._validate_init(message)


@pytest.mark.parametrize("port", [True, 0, 65536, 1.5, "18080"])
def test_init_rejects_invalid_port(tmp_path: Path, port: object) -> None:
    message = _valid_init(tmp_path)
    message["port"] = port
    with pytest.raises(ValueError, match="port"):
        higgs_worker._validate_init(message)


@pytest.mark.parametrize(
    "field", ["startup_timeout_seconds", "inference_timeout_seconds"]
)
@pytest.mark.parametrize("value", [True, 0, -1, "1"])
def test_init_rejects_invalid_timeout(
    tmp_path: Path, field: str, value: object
) -> None:
    message = _valid_init(tmp_path)
    message[field] = value
    with pytest.raises(ValueError, match=field):
        higgs_worker._validate_init(message)


def test_request_contract_has_no_raw_text_speaker_or_controls(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    _write_wav(reference)
    valid = {
        "model_input": "<|emotion:fear|>Resolved text",
        "reference_wav": str(reference),
        "output_path": str(tmp_path / "raw.wav"),
    }
    assert higgs_worker._validate_request(valid)["model_input"] == valid["model_input"]

    for forbidden in (
        "text",
        "speaker_id",
        "rate",
        "arousal",
        "coarse_affect",
        "pause_before_ms",
    ):
        with pytest.raises(ValueError, match="extra"):
            higgs_worker._validate_request({**valid, forbidden: "not allowed"})


def test_request_requires_existing_reference_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="reference_wav"):
        higgs_worker._validate_request(
            {
                "model_input": "text",
                "reference_wav": str(tmp_path / "missing.wav"),
                "output_path": str(tmp_path / "raw.wav"),
            }
        )


def test_frozen_external_payload_is_exact(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    _write_wav(reference)
    request = higgs_worker._validate_request(
        {
            "model_input": "<|emotion:anger|>Exact model input",
            "reference_wav": str(reference),
            "output_path": str(tmp_path / "raw.wav"),
        }
    )

    assert higgs_worker._speech_payload(request) == {
        "input": "<|emotion:anger|>Exact model input",
        "voice": "default",
        "response_format": "wav",
        "speed": 1.0,
        "stream": False,
        "max_new_tokens": 1024,
        "temperature": 0.8,
        "top_p": 0.8,
        "top_k": 30,
        "references": [{"audio_path": str(reference.resolve())}],
    }
    assert "text" not in higgs_worker._speech_payload(request)["references"][0]
    assert "seed" not in higgs_worker._speech_payload(request)


def test_generation_constants_match_frozen_experiment_source() -> None:
    tree = ast.parse(FROZEN_HIGGS_MAPPING.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "HTTP_LITERALS"
            for target in node.targets
        )
    )

    assert ast.literal_eval(assignment.value) == higgs_worker.FROZEN_GENERATION_FIELDS


def test_http_errors_are_closed_for_health_and_speech(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class TrackingBody(io.BytesIO):
        pass

    class HealthyResponse:
        status = 200

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"status":"healthy","running":true}'

    class Server:
        returncode = None

        def poll(self) -> None:
            return None

    health_bodies: list[TrackingBody] = []

    def health_open(*args: object, **kwargs: object) -> HealthyResponse:
        if len(health_bodies) < 25:
            body = TrackingBody(b"temporarily unavailable")
            health_bodies.append(body)
            raise urllib.error.HTTPError("health", 503, "busy", {}, body)
        return HealthyResponse()

    monkeypatch.setattr(higgs_worker.HTTP_OPENER, "open", health_open)
    monkeypatch.setattr(higgs_worker.time, "sleep", lambda _: None)
    higgs_worker._wait_for_readiness(
        Server(),  # type: ignore[arg-type]
        {"host": "127.0.0.1", "port": 18080, "startup_timeout_seconds": 2},
    )

    reference = tmp_path / "reference.wav"
    _write_wav(reference)
    request = {
        "model_input": "resolved input",
        "reference_wav": str(reference),
        "output_path": str(tmp_path / "raw.wav"),
    }
    speech_bodies: list[TrackingBody] = []

    def speech_open(*args: object, **kwargs: object) -> None:
        body = TrackingBody(b"request failed")
        speech_bodies.append(body)
        raise urllib.error.HTTPError("speech", 500, "failed", {}, body)

    monkeypatch.setattr(higgs_worker.HTTP_OPENER, "open", speech_open)
    for _ in range(25):
        with pytest.raises(RuntimeError, match="HTTP 500"):
            higgs_worker._synthesise(
                Server(),  # type: ignore[arg-type]
                {
                    "host": "127.0.0.1",
                    "port": 18080,
                    "inference_timeout_seconds": 2,
                },
                request,
            )

    assert len(health_bodies) == 25
    assert all(body.closed for body in health_bodies)
    assert len(speech_bodies) == 25
    assert all(body.closed for body in speech_bodies)


def test_readiness_does_not_sleep_after_request_exhausts_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Server:
        returncode = None

        def poll(self) -> None:
            return None

    clock = [0.0]
    sleeps: list[float] = []

    def failed_open(*args: object, **kwargs: object) -> None:
        clock[0] = 2.0
        raise urllib.error.URLError("transient")

    monkeypatch.setattr(higgs_worker.HTTP_OPENER, "open", failed_open)
    monkeypatch.setattr(higgs_worker.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(higgs_worker.time, "sleep", sleeps.append)

    with pytest.raises(TimeoutError, match="did not become ready"):
        higgs_worker._wait_for_readiness(
            Server(),  # type: ignore[arg-type]
            {"host": "127.0.0.1", "port": 18080, "startup_timeout_seconds": 1},
        )

    assert sleeps == []


def test_ready_waits_for_health_and_server_logs_do_not_pollute_stdout(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(
        harnesses,
        tmp_path,
        {"startup_delay": 0.3, "log_bytes": 300_000},
    )
    started = time.monotonic()

    response = harness.start()

    assert response == {"status": "ready"}
    assert time.monotonic() - started >= 0.2
    assert any("server-log-" in line for line in harness.stderr)


def test_many_health_http_errors_close_responses_then_become_ready(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(harnesses, tmp_path, {"health_http_errors": 20})
    harness.init["startup_timeout_seconds"] = 8

    assert harness.start(timeout=10) == {"status": "ready"}
    assert (harness.model_dir / "health-count.txt").read_text() == "21"
    assert len(list(Path(f"/proc/{harness.process.pid}/fd").iterdir())) < 16


def test_flaky_health_responses_retry_until_valid_readiness(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(
        harnesses,
        tmp_path,
        {"health_sequence": ["incomplete", "malformed", "reset"]},
    )

    assert harness.start() == {"status": "ready"}
    assert (harness.model_dir / "health-count.txt").read_text() == "4"


def test_server_stdin_is_devnull_and_worker_protocol_remains_owned(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(harnesses, tmp_path, {"read_stdin": True})

    assert harness.start() == {"status": "ready"}
    assert (harness.model_dir / "server-stdin.txt").read_text() == ""
    assert harness.request(tmp_path / "raw.wav")["status"] == "ok"


def test_proxy_environment_cannot_intercept_loopback_health_or_speech(
    tmp_path: Path,
    harnesses: list[WorkerHarness],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unusable_proxy = f"http://127.0.0.1:{_free_port()}"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(name, unusable_proxy)
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    harness = _harness(harnesses, tmp_path)

    assert harness.start() == {"status": "ready"}
    assert harness.request(tmp_path / "raw.wav")["status"] == "ok"


def test_success_uses_exact_http_request_and_reports_observed_wav(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(harnesses, tmp_path, {"sample_rate": 16000, "channels": 2})
    assert harness.start() == {"status": "ready"}
    output = tmp_path / "nested" / "raw.wav"

    response = harness.request(output, "<|prosody:expressive_low|>Model input")
    record = harness.requests()[0]

    assert response == {
        "status": "ok",
        "output_path": str(output.resolve()),
        "sample_rate": 16000,
        "channels": 2,
        "samples": 240,
    }
    assert output.is_file()
    assert not output.with_name("raw.wav.part").exists()
    assert record["payload"] == {
        "input": "<|prosody:expressive_low|>Model input",
        **higgs_worker.FROZEN_GENERATION_FIELDS,
        "references": [{"audio_path": str(harness.reference.resolve())}],
    }
    assert record["content_type"].startswith("application/json")
    assert record["accept"] == "audio/wav"


def test_startup_timeout_reports_error_and_cleans_server(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(harnesses, tmp_path, {"never_ready": True})
    harness.init["startup_timeout_seconds"] = 0.35
    harness.send(harness.init)

    response = harness.response()
    pid = harness.server_pid()

    assert response["status"] == "error"
    assert "TimeoutError" in response["message"]
    assert harness.process.wait(timeout=3) == 1
    _wait_stopped(pid)
    assert (harness.model_dir / "terminated.txt").is_file()


def test_server_exit_before_readiness_is_clear_and_fatal(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(harnesses, tmp_path, {"exit_before_ready": True})
    response = harness.start()

    assert response["status"] == "error"
    assert "exited before readiness" in response["message"]
    assert harness.process.wait(timeout=3) == 1


def test_initialization_validation_failure_exits_nonzero(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(harnesses, tmp_path)
    harness.init["host"] = "8.8.8.8"
    harness.send(harness.init)

    response = harness.response()

    assert response["status"] == "error"
    assert "loopback" in response["message"]
    assert harness.process.wait(timeout=3) == 1
    assert not (harness.model_dir / "server.pid").exists()


def test_occupied_port_fails_before_launch_without_contacting_decoy(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    requests: list[str] = []

    class DecoyHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return None

        def do_GET(self) -> None:
            requests.append(f"GET {self.path}")
            body = b'{"status":"healthy","running":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            requests.append(f"POST {self.path}")
            self.send_error(500)

    decoy = ThreadingHTTPServer((higgs_worker.DEFAULT_HOST, 0), DecoyHandler)
    thread = threading.Thread(target=decoy.serve_forever, daemon=True)
    thread.start()
    try:
        harness = _harness(harnesses, tmp_path)
        harness.init["port"] = decoy.server_address[1]

        response = harness.start()

        assert response["status"] == "error"
        assert "cannot own loopback port" in response["message"]
        assert harness.process.wait(timeout=3) == 1
        assert not (harness.model_dir / "server.pid").exists()
        assert not (harness.model_dir / "launch-count.txt").exists()
        assert requests == []
    finally:
        decoy.shutdown()
        decoy.server_close()
        thread.join(timeout=2)


def test_http_error_is_recoverable_and_later_request_reuses_server(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(harnesses, tmp_path, {"fail_first": True})
    assert harness.start() == {"status": "ready"}
    pid = harness.server_pid()

    failed = harness.request(tmp_path / "failed.wav")
    succeeded = harness.request(tmp_path / "succeeded.wav")

    assert failed["status"] == "error"
    assert "HTTP 500" in failed["message"]
    assert "fatal" not in failed
    assert succeeded["status"] == "ok"
    assert harness.server_pid() == pid
    assert (harness.model_dir / "launch-count.txt").read_text() == "1"


def test_cuda_oom_http_500_retries_once_then_publishes_wav(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(
        harnesses,
        tmp_path,
        {"fail_first": True, "fail_body": "CUDA   Out\nOf Memory"},
    )
    assert harness.start() == {"status": "ready"}
    pid = harness.server_pid()
    output = tmp_path / "raw.wav"

    response = harness.request(output)

    assert response["status"] == "ok"
    assert output.is_file()
    assert not output.with_name("raw.wav.part").exists()
    assert len(harness.requests()) == 2
    assert {record["pid"] for record in harness.requests()} == {pid}
    assert (harness.model_dir / "launch-count.txt").read_text() == "1"


def test_cuda_oom_http_500_stops_after_two_attempts_and_keeps_server(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(
        harnesses,
        tmp_path,
        {"fail_requests": 2, "fail_body": "CUDA out of memory"},
    )
    assert harness.start() == {"status": "ready"}
    pid = harness.server_pid()
    output = tmp_path / "failed.wav"

    failed = harness.request(output)

    assert failed["status"] == "error"
    assert "HTTP 500" in failed["message"]
    assert "fatal" not in failed
    assert len(harness.requests()) == 2
    assert not output.exists()
    assert not output.with_name("failed.wav.part").exists()
    assert harness.request(tmp_path / "later.wav")["status"] == "ok"
    assert harness.server_pid() == pid
    assert (harness.model_dir / "launch-count.txt").read_text() == "1"


@pytest.mark.parametrize(
    "status,body", [(500, "other error"), (400, "CUDA out of memory")]
)
def test_only_cuda_oom_http_500_is_retried(
    tmp_path: Path, harnesses: list[WorkerHarness], status: int, body: str
) -> None:
    harness = _harness(
        harnesses,
        tmp_path,
        {"fail_first": True, "fail_status": status, "fail_body": body},
    )
    assert harness.start() == {"status": "ready"}

    failed = harness.request(tmp_path / "failed.wav")

    assert failed["status"] == "error"
    assert f"HTTP {status}" in failed["message"]
    assert len(harness.requests()) == 1


@pytest.mark.parametrize(
    "failure", [TimeoutError("slow"), urllib.error.URLError("network")]
)
def test_speech_transport_failure_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    reference = tmp_path / "reference.wav"
    _write_wav(reference)
    output = tmp_path / "raw.wav"
    requests: list[object] = []

    class Server:
        returncode = None

        def poll(self) -> None:
            return None

    def fail(request: object, **_kwargs: object) -> None:
        requests.append(request)
        raise failure

    monkeypatch.setattr(higgs_worker.HTTP_OPENER, "open", fail)
    with pytest.raises(RuntimeError, match="transport failed"):
        higgs_worker._synthesise(
            Server(),  # type: ignore[arg-type]
            {"host": "127.0.0.1", "port": 18080, "inference_timeout_seconds": 2},
            {
                "model_input": "text",
                "reference_wav": str(reference),
                "output_path": str(output),
            },
        )
    assert len(requests) == 1
    assert not output.exists()
    assert not output.with_name("raw.wav.part").exists()


def test_server_death_after_oom_prevents_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "reference.wav"
    _write_wav(reference)
    output = tmp_path / "raw.wav"
    requests: list[object] = []

    class Server:
        returncode = 9
        checks = 0

        def poll(self) -> int | None:
            self.checks += 1
            return None if self.checks == 1 else 9

    def oom(request: object, **_kwargs: object) -> None:
        requests.append(request)
        raise urllib.error.HTTPError(
            "speech", 500, "error", {}, io.BytesIO(b"CUDA out of memory")
        )

    monkeypatch.setattr(higgs_worker.HTTP_OPENER, "open", oom)
    with pytest.raises(higgs_worker.FatalServerError, match="exited unexpectedly"):
        higgs_worker._synthesise(
            Server(),  # type: ignore[arg-type]
            {"host": "127.0.0.1", "port": 18080, "inference_timeout_seconds": 2},
            {
                "model_input": "text",
                "reference_wav": str(reference),
                "output_path": str(output),
            },
        )
    assert len(requests) == 1
    assert not output.exists()


def test_earlier_wav_survives_later_cuda_oom_retry(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(
        harnesses,
        tmp_path,
        {"fail_on_requests": [2, 3], "fail_body": "CUDA out of memory"},
    )
    assert harness.start() == {"status": "ready"}
    earlier = tmp_path / "turn_001.wav"
    later = tmp_path / "turn_002.wav"
    assert harness.request(earlier)["status"] == "ok"
    original = earlier.read_bytes()

    failed = harness.request(later)

    assert failed["status"] == "error"
    assert len(harness.requests()) == 3
    assert earlier.read_bytes() == original
    assert not later.exists()
    assert not later.with_name("turn_002.wav.part").exists()


def test_many_speech_http_errors_close_responses_and_reuse_server(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(harnesses, tmp_path, {"fail_requests": 40})
    assert harness.start() == {"status": "ready"}
    pid = harness.server_pid()
    baseline_fds = len(list(Path(f"/proc/{harness.process.pid}/fd").iterdir()))

    failures = [harness.request(tmp_path / "failed.wav") for _ in range(40)]
    after_failure_fds = len(list(Path(f"/proc/{harness.process.pid}/fd").iterdir()))
    succeeded = harness.request(tmp_path / "succeeded.wav")

    assert all(response["status"] == "error" for response in failures)
    assert after_failure_fds <= baseline_fds + 2
    assert succeeded["status"] == "ok"
    assert harness.server_pid() == pid
    assert len(harness.requests()) == 41


def test_malformed_wav_is_recoverable_atomic_and_preserves_existing_output(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(harnesses, tmp_path, {"malformed_first": True})
    assert harness.start() == {"status": "ready"}
    output = tmp_path / "raw.wav"
    output.write_bytes(b"existing")

    failed = harness.request(output)
    assert len(harness.requests()) == 1
    assert output.read_bytes() == b"existing"
    assert not output.with_name("raw.wav.part").exists()
    succeeded = harness.request(output)

    assert failed["status"] == "error"
    assert "readable WAV" in failed["message"]
    assert succeeded["status"] == "ok"
    assert output.read_bytes() != b"existing"
    assert not output.with_name("raw.wav.part").exists()


def test_wrong_content_type_is_a_recoverable_request_error(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(harnesses, tmp_path, {"wrong_content_type": True})
    assert harness.start() == {"status": "ready"}

    output = tmp_path / "raw.wav"
    response = harness.request(output)

    assert response["status"] == "error"
    assert "audio/wav" in response["message"]
    assert len(harness.requests()) == 1
    assert not output.exists()
    assert not output.with_name("raw.wav.part").exists()
    assert harness.process.poll() is None


def test_malformed_json_line_is_recoverable(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(harnesses, tmp_path)
    assert harness.start() == {"status": "ready"}
    assert harness.process.stdin is not None
    harness.process.stdin.write("{not json}\n")
    harness.process.stdin.flush()

    failed = harness.response()
    succeeded = harness.request(tmp_path / "raw.wav")

    assert failed["status"] == "error"
    assert "JSONDecodeError" in failed["message"]
    assert succeeded["status"] == "ok"


def test_detected_server_death_is_fatal(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(harnesses, tmp_path, {"exit_on_request": True})
    assert harness.start() == {"status": "ready"}

    response = harness.request(tmp_path / "raw.wav")

    assert response["status"] == "error"
    assert response["fatal"] is True
    assert "server exited" in response["message"]
    assert harness.process.wait(timeout=3) == 1


def test_multiple_successes_reuse_one_server_pid(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(harnesses, tmp_path)
    assert harness.start() == {"status": "ready"}
    pid = harness.server_pid()

    responses = [harness.request(tmp_path / f"raw-{index}.wav") for index in range(3)]

    assert all(response["status"] == "ok" for response in responses)
    assert {record["pid"] for record in harness.requests()} == {pid}
    assert (harness.model_dir / "launch-count.txt").read_text() == "1"


def test_stdin_eof_cleans_server_process_group_and_stdout_is_protocol_only(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(harnesses, tmp_path)
    assert harness.start() == {"status": "ready"}
    pid = harness.server_pid()

    harness.close_stdin()

    assert harness.process.wait(timeout=3) == 0
    _wait_stopped(pid)
    assert (harness.model_dir / "terminated.txt").is_file()
    assert harness.process.stdout is not None
    assert harness.process.stdout.read() == ""


def test_sigterm_enters_cleanup_path(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(harnesses, tmp_path)
    assert harness.start() == {"status": "ready"}
    pid = harness.server_pid()

    harness.process.send_signal(signal.SIGTERM)

    assert harness.process.wait(timeout=3) == 130
    _wait_stopped(pid)
    assert (harness.model_dir / "terminated.txt").is_file()


def test_harness_cleanup_reaps_fake_server_after_worker_is_killed(
    tmp_path: Path, harnesses: list[WorkerHarness]
) -> None:
    harness = _harness(harnesses, tmp_path)
    assert harness.start() == {"status": "ready"}
    pid = harness.server_pid()

    harness.process.kill()
    assert harness.process.wait(timeout=3) == -signal.SIGKILL
    assert _process_exists(pid)

    harness.cleanup()
    _wait_stopped(pid)


def test_cleanup_escalates_from_process_group_sigterm_to_sigkill(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Server:
        pid = 1234

        def __init__(self) -> None:
            self.waits = 0

        def poll(self) -> None:
            return None

        def wait(self, timeout: float) -> int:
            self.waits += 1
            raise subprocess.TimeoutExpired("fake", timeout)

    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        higgs_worker.os,
        "killpg",
        lambda pid, signum: signals.append((pid, signum)),
    )

    higgs_worker._stop_server(Server())  # type: ignore[arg-type]

    assert signals == [
        (1234, signal.SIGTERM),
        (1234, signal.SIGKILL),
    ]
    assert "timed out after SIGKILL" in capsys.readouterr().err


def test_worker_imports_only_standard_library_and_never_names_gpu_runtime() -> None:
    source = WORKER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert imports.isdisjoint({"torch", "sglang", "sglang_omni", "higgs"})
    assert "nvidia-smi" not in source
    assert "torch.cuda" not in source
