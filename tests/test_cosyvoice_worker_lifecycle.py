"""Mocked CosyVoice worker lifecycle tests; no real worker process is started."""

import io
import subprocess
from collections import deque

import pytest

from tts5703 import tts_engine


class FakeWorker:
    """Minimal Popen stand-in whose stdout is already at end of stream."""

    def __init__(
        self,
        events: list[str],
        *,
        terminate_hangs: bool = False,
        kill_error: OSError | None = None,
    ) -> None:
        self.events = events
        self.terminate_hangs = terminate_hangs
        self.kill_error = kill_error
        self.pid = 4242
        self.returncode: int | None = None
        self.stdin = io.StringIO()
        self.stdout = io.StringIO("")
        self.stderr = None
        self._cosyvoice_stderr_tail = deque(["CUDA out of memory"])

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.events.append("terminate")
        if not self.terminate_hangs:
            self.returncode = -15

    def kill(self) -> None:
        self.events.append("kill")
        if self.kill_error is not None:
            # The process won the race and exited on its own before the kill.
            self.returncode = -15
            raise self.kill_error
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        self.events.append("wait")
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd="cosyvoice-worker", timeout=timeout)
        return self.returncode


class CacheSpy:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def cache_clear(self) -> None:
        self.events.append("cache_clear")


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    recorded: list[str] = []
    monkeypatch.setattr(tts_engine, "_get_cosyvoice_worker", CacheSpy(recorded))
    return recorded


def _request(worker: FakeWorker) -> str:
    with pytest.raises(
        RuntimeError, match="CosyVoice worker closed its output"
    ) as raised:
        tts_engine._cosyvoice_request(worker, {"text": "Hello"})
    return str(raised.value)


def test_unexpected_stdout_close_terminates_and_reaps_before_clearing_cache(
    events: list[str],
) -> None:
    worker = FakeWorker(events)

    message = _request(worker)

    assert "CUDA out of memory" in message
    assert worker.returncode is not None, "worker must not be left running"
    assert "kill" not in events
    assert events.index("terminate") < events.index("cache_clear")
    assert events.index("wait", events.index("terminate")) < events.index("cache_clear")
    assert events.count("cache_clear") == 1


def test_worker_ignoring_terminate_is_killed_and_reaped(events: list[str]) -> None:
    worker = FakeWorker(events, terminate_hangs=True)

    _request(worker)

    assert worker.returncode == -9
    assert events.index("kill") < events.index("cache_clear")
    assert events[-1] == "cache_clear"


@pytest.mark.parametrize(
    "kill_error",
    [ProcessLookupError(3, "No such process"), OSError(1, "Operation not permitted")],
    ids=["process_lookup_error", "os_error"],
)
def test_kill_race_after_terminate_timeout_is_survived(
    events: list[str], kill_error: OSError
) -> None:
    worker = FakeWorker(events, terminate_hangs=True, kill_error=kill_error)

    message = _request(worker)

    assert "CUDA out of memory" in message
    assert worker.returncode is not None, "worker must still be reaped"
    assert events.index("kill") < events.index("cache_clear")
    assert events[-1] == "cache_clear", "cache clearing must still be reached"
    # terminate -> wait -> kill -> wait ordering is preserved.
    assert [event for event in events if event != "cache_clear"] == [
        "wait",
        "terminate",
        "wait",
        "kill",
        "wait",
    ]


def test_kill_race_leaves_cleanup_helper_without_raising(events: list[str]) -> None:
    worker = FakeWorker(
        events,
        terminate_hangs=True,
        kill_error=ProcessLookupError(3, "No such process"),
    )

    assert tts_engine._terminate_cosyvoice_worker(worker) is None
    assert worker.returncode == -15


def test_stdin_is_closed_so_the_worker_stops_waiting_for_input(
    events: list[str],
) -> None:
    worker = FakeWorker(events)

    _request(worker)

    assert worker.stdin.closed


def test_already_exited_worker_is_not_terminated_again(events: list[str]) -> None:
    worker = FakeWorker(events)
    worker.returncode = 1

    with pytest.raises(RuntimeError, match="CosyVoice worker exited unexpectedly"):
        tts_engine._cosyvoice_request(worker, {"text": "Hello"})

    assert "terminate" not in events
    assert "kill" not in events
    assert events == ["cache_clear"]


def test_terminate_helper_leaves_exited_worker_untouched(events: list[str]) -> None:
    worker = FakeWorker(events)
    worker.returncode = 0

    tts_engine._terminate_cosyvoice_worker(worker)

    assert events == []
