"""Offline parent lifecycle, dispatch, rate, and pipeline tests for Higgs."""

from __future__ import annotations

import asyncio
import copy
import io
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

from tts5703 import pipeline, tts_engine
from tts5703.config import load_config
from tts5703.validate import NormalizedTurn

CONFIG_PATH = Path("config/config.yaml")

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
    tts_engine._shutdown_higgs_worker()
    yield
    tts_engine._shutdown_higgs_worker()


def test_parent_starts_once_reuses_worker_and_drains_large_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    script, model_dir = _fake_parent_worker(
        tmp_path, {"stderr_lines": 300, "stderr_width": 10000}
    )
    monkeypatch.setattr(tts_engine, "_HIGGS_WORKER_SCRIPT", script)
    caplog.set_level(logging.DEBUG, logger="tts5703.tts_engine")
    proc = tts_engine._get_higgs_worker(*_worker_args(model_dir))
    try:
        assert tts_engine._get_higgs_worker(*_worker_args(model_dir)) is proc
        assert (model_dir / "parent-worker-launches.txt").read_text() == "1"
        assert proc.args == [sys.executable, str(script)]
        deadline = time.monotonic() + 2
        tail = proc._higgs_stderr_tail
        while time.monotonic() < deadline and not any(
            line.startswith("diagnostic-299-") for line in tail
        ):
            time.sleep(0.01)
        assert len(tail) <= tts_engine._HIGGS_STDERR_TAIL_LINES
        assert max(map(len, tail)) <= tts_engine._HIGGS_STDERR_FRAGMENT_CHARS + 1
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
        tts_engine._discard_higgs_worker(proc)
        _wait_stopped(pid)


def test_non_utf8_stderr_does_not_stop_drain_or_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, model_dir = _fake_parent_worker(
        tmp_path,
        {"invalid_stderr": True, "stderr_lines": 60, "stderr_width": 1500},
    )
    monkeypatch.setattr(tts_engine, "_HIGGS_WORKER_SCRIPT", script)
    proc = tts_engine._get_higgs_worker(*_worker_args(model_dir))
    output_path = tmp_path / "non-utf8-output.wav"
    try:
        response = tts_engine._higgs_request(
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
        tts_engine._discard_higgs_worker(proc)
        _wait_stopped(pid)


def test_drain_setup_failure_after_popen_reaps_uncached_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, model_dir = _fake_parent_worker(tmp_path, {})
    monkeypatch.setattr(tts_engine, "_HIGGS_WORKER_SCRIPT", script)
    real_popen = subprocess.Popen
    created: list[subprocess.Popen[str]] = []

    def capture_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        proc = real_popen(*args, **kwargs)
        created.append(proc)
        return proc

    def fail_drain(proc: subprocess.Popen[str]) -> deque[str]:
        raise RuntimeError("synthetic stderr drain setup failure")

    monkeypatch.setattr(tts_engine.subprocess, "Popen", capture_popen)
    monkeypatch.setattr(tts_engine, "_drain_higgs_stderr", fail_drain)

    with pytest.raises(RuntimeError, match="synthetic stderr drain setup failure"):
        tts_engine._get_higgs_worker(*_worker_args(model_dir))

    assert len(created) == 1
    proc = created[0]
    assert proc.poll() is not None
    _wait_stopped(proc.pid)
    assert tts_engine._higgs_worker_cache is None


def test_changing_runtime_config_reaps_worker_before_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, first_model = _fake_parent_worker(tmp_path, {})
    second_model = tmp_path / "other-model"
    second_model.mkdir()
    (second_model / "parent-worker-behavior.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(tts_engine, "_HIGGS_WORKER_SCRIPT", script)
    first = tts_engine._get_higgs_worker(*_worker_args(first_model))
    first_pid = first.pid

    second = tts_engine._get_higgs_worker(*_worker_args(second_model))
    try:
        assert second is not first
        assert first.poll() is not None
        _wait_stopped(first_pid)
        assert (first_model / "parent-worker-launches.txt").read_text() == "1"
        assert (second_model / "parent-worker-launches.txt").read_text() == "1"
    finally:
        tts_engine._discard_higgs_worker(second)


def test_startup_failure_includes_bounded_stderr_and_reaps_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, model_dir = _fake_parent_worker(
        tmp_path,
        {"startup": "error", "stderr_lines": 150, "stderr_width": 1000},
    )
    monkeypatch.setattr(tts_engine, "_HIGGS_WORKER_SCRIPT", script)

    with pytest.raises(RuntimeError, match="synthetic startup failure") as raised:
        tts_engine._get_higgs_worker(*_worker_args(model_dir))

    assert "diagnostic-149-" in str(raised.value)
    assert tts_engine._higgs_worker_cache is None
    _wait_stopped(int((model_dir / "parent-worker.pid").read_text()))


def test_invalid_startup_json_includes_stderr_and_reaps_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, model_dir = _fake_parent_worker(
        tmp_path, {"startup": "invalid", "stderr_lines": 1}
    )
    monkeypatch.setattr(tts_engine, "_HIGGS_WORKER_SCRIPT", script)

    with pytest.raises(RuntimeError, match="invalid startup data") as raised:
        tts_engine._get_higgs_worker(*_worker_args(model_dir))

    assert "diagnostic-0-" in str(raised.value)
    assert tts_engine._higgs_worker_cache is None
    _wait_stopped(int((model_dir / "parent-worker.pid").read_text()))


def test_already_exited_worker_clears_cache_with_stderr_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, model_dir = _fake_parent_worker(tmp_path, {"exit_after_ready": True})
    monkeypatch.setattr(tts_engine, "_HIGGS_WORKER_SCRIPT", script)
    proc = tts_engine._get_higgs_worker(*_worker_args(model_dir))
    assert proc.wait(timeout=3) == 7
    deadline = time.monotonic() + 1
    while (
        time.monotonic() < deadline
        and "worker-exited-after-ready" not in proc._higgs_stderr_tail
    ):
        time.sleep(0.01)

    with pytest.raises(RuntimeError, match="worker-exited-after-ready"):
        tts_engine._higgs_request(proc, {"model_input": "unused"})

    assert tts_engine._higgs_worker_cache is None


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
    monkeypatch.setattr(tts_engine, "_HIGGS_WORKER_SCRIPT", script)
    proc = tts_engine._get_higgs_worker(*_worker_args(model_dir))
    pid = proc.pid

    with pytest.raises(RuntimeError, match=message) as raised:
        tts_engine._higgs_request(
            proc,
            {
                "model_input": "resolved",
                "reference_wav": "/reference.wav",
                "output_path": str(tmp_path / "raw.wav"),
            },
        )

    if mode == "fatal":
        assert "runtime-fatal-diagnostic" in str(raised.value)
    assert tts_engine._higgs_worker_cache is None
    _wait_stopped(pid)


def test_recoverable_error_keeps_worker_for_later_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, model_dir = _fake_parent_worker(tmp_path, {"request": "recoverable_first"})
    monkeypatch.setattr(tts_engine, "_HIGGS_WORKER_SCRIPT", script)
    proc = tts_engine._get_higgs_worker(*_worker_args(model_dir))
    request = {
        "model_input": "resolved",
        "reference_wav": "/reference.wav",
        "output_path": str(tmp_path / "raw.wav"),
    }
    try:
        with pytest.raises(RuntimeError, match="synthetic recoverable"):
            tts_engine._higgs_request(proc, request)

        response = tts_engine._higgs_request(proc, request)

        assert response["status"] == "ok"
        assert proc.poll() is None
        assert tts_engine._get_higgs_worker(*_worker_args(model_dir)) is proc
        assert (model_dir / "parent-worker-launches.txt").read_text() == "1"
    finally:
        tts_engine._discard_higgs_worker(proc)


def _turn(
    *,
    turn_id: int = 1,
    speaker: str = "spk_001",
    rate: str = "normal",
    arousal: str | None = "low",
    coarse_affect: str | None = "warm",
) -> NormalizedTurn:
    return NormalizedTurn(
        turn_id=turn_id,
        speaker=speaker,
        text="This raw turn text must be transformed.",
        label="normal",
        rate=rate,
        pause_before_ms=25,
        pause_after_ms=50,
        arousal=arousal,
        coarse_affect=coarse_affect,
    )


def _runtime_config(tmp_path: Path) -> dict[str, Any]:
    server = tmp_path / "sgl-omni"
    server.write_text("fake", encoding="utf-8")
    server.chmod(0o755)
    model_dir = tmp_path / "higgs-model"
    model_dir.mkdir()
    reference = tmp_path / "reference.wav"
    _write_wav(reference)
    return {
        "tts": {
            "engine": "higgs",
            "higgs": {
                "server_executable": str(server),
                "model_dir": str(model_dir),
                "voice_map": {"spk_001": {"reference_wav": str(reference)}},
            },
        }
    }


class ProtocolWorker:
    def __init__(self, response: dict[str, Any]) -> None:
        self.returncode = None
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(json.dumps(response) + "\n")
        self._higgs_stderr_tail: deque[str] = deque()

    def poll(self) -> None:
        return None


def _successful_request(captured: list[dict[str, Any]]):
    def request(worker: object, payload: dict[str, Any]) -> dict[str, Any]:
        captured.append(payload)
        output = Path(payload["output_path"])
        _write_wav(output)
        return {
            "status": "ok",
            "output_path": str(output.resolve()),
            "sample_rate": 24000,
            "channels": 1,
            "samples": 4800,
        }

    return request


def test_dispatch_sends_only_resolved_model_input_and_absolute_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _runtime_config(tmp_path)
    captured: list[dict[str, Any]] = []
    worker_args: list[tuple[Any, ...]] = []

    def get_worker(*args: Any) -> object:
        worker_args.append(args)
        return object()

    monkeypatch.setattr(tts_engine, "_get_higgs_worker", get_worker)
    monkeypatch.setattr(tts_engine, "_higgs_request", _successful_request(captured))
    monkeypatch.setattr(
        tts_engine.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("normal rate must not run FFmpeg"),
    )

    final = asyncio.run(tts_engine.synthesize_turn(_turn(), tmp_path, config))

    assert final == tmp_path / "turn_001.wav"
    assert final.is_file()
    assert worker_args == [
        (
            str((tmp_path / "sgl-omni").resolve()),
            str((tmp_path / "higgs-model").resolve()),
            "127.0.0.1",
            18080,
            900,
            300,
        )
    ]
    assert captured == [
        {
            "model_input": (
                "<|emotion:contentment|>This raw turn text must be transformed."
            ),
            "reference_wav": str((tmp_path / "reference.wav").resolve()),
            "output_path": str((tmp_path / "turn_001.higgs_raw.wav").resolve()),
        }
    ]
    assert not (tmp_path / "turn_001.higgs_raw.wav").exists()
    assert set(captured[0]) == {"model_input", "reference_wav", "output_path"}


def test_success_response_rejects_wrong_or_missing_raw_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker: Any = ProtocolWorker({"status": "ok"})
    discarded: list[object] = []
    monkeypatch.setattr(tts_engine, "_discard_higgs_worker", discarded.append)
    expected = tmp_path / "expected.higgs_raw.wav"

    with pytest.raises(RuntimeError, match="has no output_path"):
        tts_engine._validate_higgs_success(worker, {"status": "ok"}, expected)

    with pytest.raises(RuntimeError, match="unexpected output_path"):
        tts_engine._validate_higgs_success(
            worker,
            {"status": "ok", "output_path": str(tmp_path / "other.wav")},
            expected,
        )

    with pytest.raises(RuntimeError, match="did not produce a non-empty WAV"):
        tts_engine._validate_higgs_success(
            worker,
            {"status": "ok", "output_path": str(expected)},
            expected,
        )

    assert discarded == [worker, worker, worker]


@pytest.mark.parametrize(("rate", "factor"), [("slow", "0.85"), ("fast", "1.15")])
def test_rate_postprocess_uses_exact_resolver_factor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rate: str,
    factor: str,
) -> None:
    config = _runtime_config(tmp_path)
    config["tts"]["higgs"]["ffmpeg_bin"] = "/fake/ffmpeg"
    captured_requests: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    monkeypatch.setattr(tts_engine, "_get_higgs_worker", lambda *args: object())
    monkeypatch.setattr(
        tts_engine, "_higgs_request", _successful_request(captured_requests)
    )

    def fake_ffmpeg(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[-1]).write_bytes(Path(command[3]).read_bytes())
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(tts_engine.subprocess, "run", fake_ffmpeg)

    final = asyncio.run(tts_engine.synthesize_turn(_turn(rate=rate), tmp_path, config))

    assert commands == [
        [
            "/fake/ffmpeg",
            "-y",
            "-i",
            str(tmp_path / "turn_001.higgs_raw.wav"),
            "-af",
            f"atempo={factor}",
            str(tmp_path / "turn_001.higgs_processed.part.wav"),
        ]
    ]
    assert final.is_file()
    assert not (tmp_path / "turn_001.higgs_raw.wav").exists()
    assert not (tmp_path / "turn_001.higgs_processed.part.wav").exists()
    assert "rate" not in captured_requests[0]


def test_ffmpeg_failure_preserves_old_final_and_retains_raw_for_diagnosis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _runtime_config(tmp_path)
    final = tmp_path / "turn_001.wav"
    final.write_bytes(b"historical-final")
    monkeypatch.setattr(tts_engine, "_get_higgs_worker", lambda *args: object())
    monkeypatch.setattr(tts_engine, "_higgs_request", _successful_request([]))
    monkeypatch.setattr(
        tts_engine.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 7, "", "synthetic ffmpeg failure"
        ),
    )

    with pytest.raises(RuntimeError, match="synthetic ffmpeg failure"):
        asyncio.run(tts_engine.synthesize_turn(_turn(rate="slow"), tmp_path, config))

    assert final.read_bytes() == b"historical-final"
    assert (tmp_path / "turn_001.higgs_raw.wav").is_file()
    assert not (tmp_path / "turn_001.higgs_processed.part.wav").exists()


def test_failed_current_request_never_uses_stale_final_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _runtime_config(tmp_path)
    final = tmp_path / "turn_001.wav"
    final.write_bytes(b"historical-final")
    worker = ProtocolWorker({"status": "error", "message": "new request failed"})
    monkeypatch.setattr(tts_engine, "_get_higgs_worker", lambda *args: worker)

    with pytest.raises(RuntimeError, match="new request failed"):
        asyncio.run(tts_engine.synthesize_turn(_turn(), tmp_path, config))

    assert final.read_bytes() == b"historical-final"
    assert not (tmp_path / "turn_001.higgs_raw.wav").exists()


def test_pipeline_reports_failure_despite_stale_final_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_config = _runtime_config(tmp_path)
    config = copy.deepcopy(load_config(CONFIG_PATH))
    config["tts"]["engine"] = "higgs"
    config["tts"]["higgs"] = runtime_config["tts"]["higgs"]
    worker = ProtocolWorker({"status": "error", "message": "new request failed"})
    monkeypatch.setattr(tts_engine, "_get_higgs_worker", lambda *args: worker)
    dialogue = {
        "schema_version": "0.2",
        "dialogue_id": "stale_higgs",
        "turns": [
            {
                "turn_id": 1,
                "speaker": "spk_001",
                "text": "This request must fail.",
                "label": "normal",
                "acoustic_spec": {"rate": "normal"},
            }
        ],
    }
    dialogue_path = tmp_path / "stale-dialogue.json"
    dialogue_path.write_text(json.dumps(dialogue), encoding="utf-8")
    output_root = tmp_path / "output"
    out_dir = output_root / "stale_higgs"
    out_dir.mkdir(parents=True)
    final = out_dir / "turn_001.wav"
    final.write_bytes(b"historical-final")

    result = asyncio.run(pipeline.run_dialogue(dialogue_path, config, output_root))

    assert result.status == "failed"
    assert result.error is not None and "new request failed" in result.error
    assert final.read_bytes() == b"historical-final"


def _write_fake_sglang(path: Path) -> None:
    path.write_text(f"#!{sys.executable}\n{FAKE_SGLANG}", encoding="utf-8")
    path.chmod(0o755)


def test_offline_higgs_pipeline_e2e_with_two_speaker_references(tmp_path: Path) -> None:
    config = copy.deepcopy(load_config(CONFIG_PATH))
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    server = tmp_path / "fake-sgl-omni"
    _write_fake_sglang(server)
    references = {}
    for speaker in ("spk_001", "spk_002"):
        reference = tmp_path / f"{speaker}.wav"
        _write_wav(reference)
        references[speaker] = {"reference_wav": str(reference)}
    port = _free_port()
    config["tts"]["engine"] = "higgs"
    config["tts"]["higgs"] = {
        "server_executable": str(server),
        "model_dir": str(model_dir),
        "host": "127.0.0.1",
        "port": port,
        "startup_timeout_seconds": 3,
        "inference_timeout_seconds": 3,
        "voice_map": references,
    }
    dialogue = {
        "schema_version": "0.2",
        "dialogue_id": "higgs_fake_e2e",
        "turns": [
            {
                "turn_id": 1,
                "speaker": "spk_001",
                "text": "Please tell me what happened.",
                "label": "normal",
                "acoustic_spec": {
                    "rate": "normal",
                    "arousal": "low",
                    "coarse_affect": "warm",
                    "pause_after_ms": 50,
                },
            },
            {
                "turn_id": 2,
                "speaker": "spk_002",
                "text": "I am worried and need help.",
                "label": "alert",
                "acoustic_spec": {
                    "rate": "normal",
                    "arousal": "high",
                    "coarse_affect": "anxious",
                },
            },
        ],
    }
    dialogue_path = tmp_path / "dialogue.json"
    dialogue_path.write_text(json.dumps(dialogue), encoding="utf-8")
    output_root = tmp_path / "output"

    result = asyncio.run(pipeline.run_dialogue(dialogue_path, config, output_root))

    server_pid = int((model_dir / "fake-sglang.pid").read_text())
    tts_engine._shutdown_higgs_worker()
    _wait_stopped(server_pid)
    assert tts_engine._higgs_worker_cache is None

    assert result.status == "success"
    assert result.qc is not None and result.qc.passed
    out_dir = output_root / "higgs_fake_e2e"
    assert (out_dir / "higgs_fake_e2e_clean.wav").is_file()
    assert (out_dir / "higgs_fake_e2e_telephone.wav").is_file()
    assert (out_dir / "turn_001.wav").is_file()
    assert (out_dir / "turn_002.wav").is_file()
    assert not list(out_dir.glob("*.higgs_raw.wav"))
    requests = _requests(model_dir, "fake-sglang-requests.jsonl")
    assert len(requests) == 2
    assert requests[0]["input"].startswith("<|emotion:contentment|>")
    assert requests[0]["references"] == [
        {"audio_path": str((tmp_path / "spk_001.wav").resolve())}
    ]
    assert requests[1]["references"] == [
        {"audio_path": str((tmp_path / "spk_002.wav").resolve())}
    ]
    assert (model_dir / "fake-sglang-launches.txt").read_text() == "1"
