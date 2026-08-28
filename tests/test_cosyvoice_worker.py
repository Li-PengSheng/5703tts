"""CosyVoice worker protocol tests using tiny fake dependencies."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

WORKER = Path("src/tts5703/cosyvoice_worker.py").resolve()


def _prepare_fake_cosyvoice(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    repo = tmp_path / "CosyVoice"
    cli = repo / "cosyvoice" / "cli"
    matcha = repo / "third_party" / "Matcha-TTS"
    model_dir = tmp_path / "model"
    cli.mkdir(parents=True)
    matcha.mkdir(parents=True)
    model_dir.mkdir()
    (repo / "cosyvoice" / "__init__.py").write_text("", encoding="utf-8")
    (cli / "__init__.py").write_text("", encoding="utf-8")
    (model_dir / "cosyvoice3.yaml").write_text("fake", encoding="utf-8")
    (cli / "cosyvoice.py").write_text(
        """
import json
from pathlib import Path

class FakeSpeech:
    shape = (1, 4)
    def detach(self): return self
    def cpu(self): return self

def record(prompt_wav, payload):
    path = Path(prompt_wav).with_name('calls.jsonl')
    with path.open('a', encoding='utf-8') as output:
        output.write(json.dumps(payload) + '\\n')

class CosyVoice3:
    sample_rate = 24000
    def __init__(self, **kwargs): pass
    def inference_zero_shot(
        self, text, prompt_text, prompt_wav, stream=False, speed=1.0
    ):
        print('fake zero-shot progress must not enter protocol stdout')
        record(prompt_wav, {
            'mode': 'zero_shot',
            'text': text,
            'prompt_text': prompt_text,
            'prompt_wav': prompt_wav,
            'stream': stream,
            'speed': speed,
        })
        yield {'tts_speech': FakeSpeech()}
    def inference_instruct2(
        self, text, instruction, prompt_wav, stream=False, speed=1.0
    ):
        print('fake instruct2 progress must not enter protocol stdout')
        record(prompt_wav, {
            'mode': 'instruct2',
            'text': text,
            'instruction': instruction,
            'prompt_wav': prompt_wav,
            'stream': stream,
            'speed': speed,
        })
        yield {'tts_speech': FakeSpeech()}
""",
        encoding="utf-8",
    )
    (repo / "torch.py").write_text(
        """
class Audio:
    shape = (1, 4)
def cat(chunks, dim): return Audio()
""",
        encoding="utf-8",
    )
    (repo / "torchaudio.py").write_text(
        """
from pathlib import Path
def save(path, audio, sample_rate, backend=None):
    Path(path).write_bytes(b'RIFF-fake-wave')
""",
        encoding="utf-8",
    )
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"RIFF-fake-prompt")
    calls = tmp_path / "calls.jsonl"
    init = {
        "repo_dir": str(repo),
        "model_dir": str(model_dir),
        "load_trt": False,
        "load_vllm": False,
        "fp16": False,
    }
    return init, prompt, calls


def _run_worker(
    init: dict[str, Any], requests: list[dict[str, Any]]
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, Any]]]:
    input_lines = [json.dumps(init), *(json.dumps(request) for request in requests)]
    result = subprocess.run(
        [sys.executable, str(WORKER)],
        input="\n".join(input_lines) + "\n",
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    return result, responses


def _zero_shot_request(prompt: Path, output: Path) -> dict[str, Any]:
    return {
        "text": "Hello",
        "prompt_text": "Reference",
        "prompt_wav": str(prompt),
        "output_path": str(output),
    }


def _read_calls(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_old_style_request_defaults_to_zero_shot_and_writes_audio(
    tmp_path: Path,
) -> None:
    init, prompt, calls_path = _prepare_fake_cosyvoice(tmp_path)
    output = tmp_path / "old_style.wav"

    result, responses = _run_worker(init, [_zero_shot_request(prompt, output)])
    calls = _read_calls(calls_path)

    assert result.returncode == 0, result.stderr
    assert responses[0] == {"status": "ready", "sample_rate": 24000}
    assert responses[1]["status"] == "ok"
    assert responses[1]["mode"] == "zero_shot"
    assert responses[1]["speed"] == 1.0
    assert calls[0]["mode"] == "zero_shot"
    assert calls[0]["speed"] == 1.0
    assert calls[0]["stream"] is False
    assert output.read_bytes() == b"RIFF-fake-wave"
    assert "fake zero-shot progress" in result.stderr


def test_zero_shot_passes_explicit_speed(tmp_path: Path) -> None:
    init, prompt, calls_path = _prepare_fake_cosyvoice(tmp_path)
    request = _zero_shot_request(prompt, tmp_path / "slow.wav")
    request.update({"mode": "zero_shot", "speed": 0.8})

    result, responses = _run_worker(init, [request])
    call = _read_calls(calls_path)[0]

    assert result.returncode == 0, result.stderr
    assert responses[1]["status"] == "ok"
    assert call["mode"] == "zero_shot"
    assert call["speed"] == 0.8


def test_instruct2_passes_instruction_and_speed(tmp_path: Path) -> None:
    init, prompt, calls_path = _prepare_fake_cosyvoice(tmp_path)
    instruction = (
        "You are a helpful assistant. Use an energetic, intense delivery."
        "<|endofprompt|>"
    )
    request = {
        "text": "Hello",
        "prompt_wav": str(prompt),
        "output_path": str(tmp_path / "instruct2.wav"),
        "mode": "instruct2",
        "speed": 1.2,
        "instruction": instruction,
    }

    result, responses = _run_worker(init, [request])
    call = _read_calls(calls_path)[0]

    assert result.returncode == 0, result.stderr
    assert responses[1]["status"] == "ok"
    assert responses[1]["mode"] == "instruct2"
    assert responses[1]["speed"] == 1.2
    assert call == {
        "mode": "instruct2",
        "text": "Hello",
        "instruction": instruction,
        "prompt_wav": str(prompt.resolve()),
        "stream": False,
        "speed": 1.2,
    }


def test_unsupported_mode_returns_clear_error(tmp_path: Path) -> None:
    init, prompt, calls_path = _prepare_fake_cosyvoice(tmp_path)
    request = _zero_shot_request(prompt, tmp_path / "unsupported.wav")
    request["mode"] = "unknown"

    result, responses = _run_worker(init, [request])

    assert result.returncode == 0, result.stderr
    assert responses[1]["status"] == "error"
    assert "unsupported CosyVoice mode: 'unknown'" in responses[1]["message"]
    assert not calls_path.exists()
