"""Offline parent-process lifecycle tests for Higgs."""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
import wave
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from tts5703.backends import higgs as higgs_backend

FAKE_PARENT_WORKER = r"""
import json
import os
import sys
import wave
from pathlib import Path

init = json.loads(sys.stdin.readline())
model_dir = Path(init["model_dir"])
behavior = json.loads((model_dir / "parent-worker-behavior.json").read_text())
(model_dir / "parent-worker-init.json").write_text(json.dumps(init))
count_path = model_dir / "parent-worker-launches.txt"
count = int(count_path.read_text() or "0") if count_path.exists() else 0
count_path.write_text(str(count + 1))
(model_dir / "parent-worker.pid").write_text(str(os.getpid()))

if behavior.get("invalid_stderr"):
    sys.stderr.buffer.write(b"invalid-utf8-\xff-diagnostic\n")
    sys.stderr.buffer.flush()
for index in range(int(behavior.get("stderr_lines", 0))):
    print(f"diagnostic-{index}-" + "x" * int(behavior.get("stderr_width", 20)), file=sys.stderr, flush=True)

startup = behavior.get("startup", "ready")
if startup == "error":
    print(json.dumps({"status": "error", "message": "synthetic startup failure"}), flush=True)
    raise SystemExit(1)
if startup == "invalid":
    print("{invalid-startup", flush=True)
    raise SystemExit(1)
print(json.dumps({"status": "ready"}), flush=True)
if behavior.get("exit_after_ready"):
    print("worker-exited-after-ready", file=sys.stderr, flush=True)
    raise SystemExit(7)

request_count = 0
for line in sys.stdin:
    request_count += 1
    request = json.loads(line)
    with (model_dir / "parent-worker-requests.jsonl").open("a") as handle:
        handle.write(json.dumps(request) + "\n")
    mode = behavior.get("request", "ok")
    if mode == "recoverable_first" and request_count == 1:
        print(json.dumps({"status": "error", "message": "synthetic recoverable"}), flush=True)
        continue
    if mode == "fatal":
        print("runtime-fatal-diagnostic", file=sys.stderr, flush=True)
        print(json.dumps({"status": "error", "message": "synthetic fatal", "fatal": True}), flush=True)
        continue
    if mode == "invalid":
        print("{invalid-response", flush=True)
        continue
    if mode == "eof":
        break
    output = Path(request["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\x00\x00" * 4800)
    print(json.dumps({
        "status": "ok",
        "output_path": str(output.resolve()),
        "sample_rate": 24000,
        "channels": 1,
        "samples": 4800,
    }), flush=True)
"""

FAKE_SGLANG = r"""
import io
import json
import os
import signal
import sys
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

args = sys.argv
model_dir = Path(args[args.index("--model-path") + 1])
host = args[args.index("--host") + 1]
port = int(args[args.index("--port") + 1])
(model_dir / "fake-sglang.pid").write_text(str(os.getpid()))
count_path = model_dir / "fake-sglang-launches.txt"
count = int(count_path.read_text() or "0") if count_path.exists() else 0
count_path.write_text(str(count + 1))

def stop(signum, frame):
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\x00\x00" * 4800)
    return output.getvalue()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(format % args, file=sys.stderr, flush=True)

    def do_GET(self):
        body = b'{"status":"healthy","running":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        with (model_dir / "fake-sglang-requests.jsonl").open("a") as handle:
            handle.write(json.dumps(request) + "\n")
        body = wav_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

ThreadingHTTPServer((host, port), Handler).serve_forever()
"""


def _write_wav(path: Path, frames: int = 4800) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\x00\x00" * frames)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


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


def _fake_parent_worker(tmp_path: Path, behavior: dict[str, Any]) -> tuple[Path, Path]:
    script = tmp_path / "fake-higgs-worker.py"
    script.write_text(FAKE_PARENT_WORKER, encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "parent-worker-behavior.json").write_text(
        json.dumps(behavior), encoding="utf-8"
    )
    return script, model_dir


def _worker_args(model_dir: Path) -> tuple[str, str, str, int, float, float]:
    return ("/fake/sgl-omni", str(model_dir), "127.0.0.1", 18080, 2, 2)


def _requests(model_dir: Path, name: str) -> list[dict[str, Any]]:
    path = model_dir / name
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(autouse=True)
def shutdown_higgs_worker() -> Any:
    higgs_backend._shutdown_worker()
    yield
    higgs_backend._shutdown_worker()


def test_parent_starts_once_reuses_worker_and_drains_large_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    script, model_dir = _fake_parent_worker(
        tmp_path, {"stderr_lines": 300, "stderr_width": 10000}
    )
    monkeypatch.setattr(higgs_backend, "_WORKER_SCRIPT", script)
    caplog.set_level(logging.DEBUG, logger="tts5703.tts_engine")
    proc = higgs_backend._get_worker(*_worker_args(model_dir))
    try:
        assert higgs_backend._get_worker(*_worker_args(model_dir)) is proc
        assert (model_dir / "parent-worker-launches.txt").read_text() == "1"
        assert proc.args == [sys.executable, str(script)]
        deadline = time.monotonic() + 2
        tail = proc._higgs_stderr_tail
        while time.monotonic() < deadline and not any(
            line.startswith("diagnostic-299-") for line in tail
        ):
            time.sleep(0.01)
        assert len(tail) <= higgs_backend._STDERR_TAIL_LINES
        assert max(map(len, tail)) <= higgs_backend._STDERR_FRAGMENT_CHARS + 1
        assert any(line.startswith("diagnostic-299-") for line in tail)
        assert "event=higgs_worker_stderr" in caplog.text
        assert json.loads((model_dir / "parent-worker-init.json").read_text()) == {
            "server_executable": "/fake/sgl-omni",
            "model_dir": str(model_dir),
            "host": "127.0.0.1",
            "port": 18080,
            "startup_timeout_seconds": 2,
            "inference_timeout_seconds": 2,
        }
    finally:
        pid = proc.pid
        higgs_backend._discard_worker(proc)
        _wait_stopped(pid)


def test_non_utf8_stderr_does_not_stop_drain_or_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, model_dir = _fake_parent_worker(
        tmp_path,
        {"invalid_stderr": True, "stderr_lines": 60, "stderr_width": 1500},
    )
    monkeypatch.setattr(higgs_backend, "_WORKER_SCRIPT", script)
    proc = higgs_backend._get_worker(*_worker_args(model_dir))
    output_path = tmp_path / "non-utf8-output.wav"
    try:
        response = higgs_backend._request(
            proc,
            {
                "model_input": "resolved",
                "reference_wav": "/reference.wav",
                "output_path": str(output_path),
            },
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not any(
            line.startswith("diagnostic-59-") for line in proc._higgs_stderr_tail
        ):
            time.sleep(0.01)

        assert proc.stderr is not None
        assert proc.stderr.encoding.lower().replace("-", "") == "utf8"
        assert proc.stderr.errors == "replace"
        assert response["status"] == "ok"
        assert output_path.is_file()
        assert any("\ufffd" in line for line in proc._higgs_stderr_tail)
        assert any(
            line.startswith("diagnostic-59-") for line in proc._higgs_stderr_tail
        )
    finally:
        pid = proc.pid
        higgs_backend._discard_worker(proc)
        _wait_stopped(pid)


def test_drain_setup_failure_after_popen_reaps_uncached_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, model_dir = _fake_parent_worker(tmp_path, {})
    monkeypatch.setattr(higgs_backend, "_WORKER_SCRIPT", script)
    real_popen = subprocess.Popen
    created: list[subprocess.Popen[str]] = []

    def capture_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        proc = real_popen(*args, **kwargs)
        created.append(proc)
        return proc

    def fail_drain(proc: subprocess.Popen[str]) -> deque[str]:
        raise RuntimeError("synthetic stderr drain setup failure")

    monkeypatch.setattr(higgs_backend.subprocess, "Popen", capture_popen)
    monkeypatch.setattr(higgs_backend, "_drain_stderr", fail_drain)

    with pytest.raises(RuntimeError, match="synthetic stderr drain setup failure"):
        higgs_backend._get_worker(*_worker_args(model_dir))

    assert len(created) == 1
    proc = created[0]
    assert proc.poll() is not None
    _wait_stopped(proc.pid)
    assert higgs_backend._worker_cache is None


def test_changing_runtime_config_reaps_worker_before_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, first_model = _fake_parent_worker(tmp_path, {})
    second_model = tmp_path / "other-model"
    second_model.mkdir()
    (second_model / "parent-worker-behavior.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(higgs_backend, "_WORKER_SCRIPT", script)
    first = higgs_backend._get_worker(*_worker_args(first_model))
    first_pid = first.pid

    second = higgs_backend._get_worker(*_worker_args(second_model))
    try:
        assert second is not first
        assert first.poll() is not None
        _wait_stopped(first_pid)
        assert (first_model / "parent-worker-launches.txt").read_text() == "1"
        assert (second_model / "parent-worker-launches.txt").read_text() == "1"
    finally:
        higgs_backend._discard_worker(second)


def test_startup_failure_includes_bounded_stderr_and_reaps_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, model_dir = _fake_parent_worker(
        tmp_path,
        {"startup": "error", "stderr_lines": 150, "stderr_width": 1000},
    )
    monkeypatch.setattr(higgs_backend, "_WORKER_SCRIPT", script)

    with pytest.raises(RuntimeError, match="synthetic startup failure") as raised:
        higgs_backend._get_worker(*_worker_args(model_dir))

    assert "diagnostic-149-" in str(raised.value)
    assert higgs_backend._worker_cache is None
    _wait_stopped(int((model_dir / "parent-worker.pid").read_text()))


def test_invalid_startup_json_includes_stderr_and_reaps_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, model_dir = _fake_parent_worker(
        tmp_path, {"startup": "invalid", "stderr_lines": 1}
    )
    monkeypatch.setattr(higgs_backend, "_WORKER_SCRIPT", script)

    with pytest.raises(RuntimeError, match="invalid startup data") as raised:
        higgs_backend._get_worker(*_worker_args(model_dir))

    assert "diagnostic-0-" in str(raised.value)
    assert higgs_backend._worker_cache is None
    _wait_stopped(int((model_dir / "parent-worker.pid").read_text()))


def test_already_exited_worker_clears_cache_with_stderr_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, model_dir = _fake_parent_worker(tmp_path, {"exit_after_ready": True})
    monkeypatch.setattr(higgs_backend, "_WORKER_SCRIPT", script)
    proc = higgs_backend._get_worker(*_worker_args(model_dir))
    assert proc.wait(timeout=3) == 7
    deadline = time.monotonic() + 1
    while (
        time.monotonic() < deadline
        and "worker-exited-after-ready" not in proc._higgs_stderr_tail
    ):
        time.sleep(0.01)

    with pytest.raises(RuntimeError, match="worker-exited-after-ready"):
        higgs_backend._request(proc, {"model_input": "unused"})

    assert higgs_backend._worker_cache is None


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("eof", "closed its output"),
        ("invalid", "protocol failed"),
        ("fatal", "fatal error"),
    ],
)
def test_unusable_worker_response_clears_cache_and_reaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    message: str,
) -> None:
    script, model_dir = _fake_parent_worker(tmp_path, {"request": mode})
    monkeypatch.setattr(higgs_backend, "_WORKER_SCRIPT", script)
    proc = higgs_backend._get_worker(*_worker_args(model_dir))
    pid = proc.pid

    with pytest.raises(RuntimeError, match=message) as raised:
        higgs_backend._request(
            proc,
            {
                "model_input": "resolved",
                "reference_wav": "/reference.wav",
                "output_path": str(tmp_path / "raw.wav"),
            },
        )

    if mode == "fatal":
        assert "runtime-fatal-diagnostic" in str(raised.value)
    assert higgs_backend._worker_cache is None
    _wait_stopped(pid)


def test_recoverable_error_keeps_worker_for_later_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, model_dir = _fake_parent_worker(tmp_path, {"request": "recoverable_first"})
    monkeypatch.setattr(higgs_backend, "_WORKER_SCRIPT", script)
    proc = higgs_backend._get_worker(*_worker_args(model_dir))
    request = {
        "model_input": "resolved",
        "reference_wav": "/reference.wav",
        "output_path": str(tmp_path / "raw.wav"),
    }
    try:
        with pytest.raises(RuntimeError, match="synthetic recoverable"):
            higgs_backend._request(proc, request)

        response = higgs_backend._request(proc, request)

        assert response["status"] == "ok"
        assert proc.poll() is None
        assert higgs_backend._get_worker(*_worker_args(model_dir)) is proc
        assert (model_dir / "parent-worker-launches.txt").read_text() == "1"
    finally:
        higgs_backend._discard_worker(proc)
