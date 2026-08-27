"""CosyVoice worker protocol tests using tiny fake dependencies."""

import json
import subprocess
import sys
from pathlib import Path

WORKER = Path("src/tts5703/cosyvoice_worker.py").resolve()


def test_worker_protocol_without_loading_real_model(tmp_path: Path) -> None:
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
class FakeSpeech:
    shape = (1, 4)
    def detach(self): return self
    def cpu(self): return self

class CosyVoice3:
    sample_rate = 24000
    def __init__(self, **kwargs): pass
    def inference_zero_shot(self, *args, **kwargs):
        print('third-party progress must not enter protocol stdout')
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
    output = tmp_path / "output.wav"

    proc = subprocess.Popen(
        [sys.executable, str(WORKER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    init = {
        "repo_dir": str(repo),
        "model_dir": str(model_dir),
        "load_trt": False,
        "load_vllm": False,
        "fp16": False,
    }
    request = {
        "text": "Hello",
        "prompt_text": "Reference",
        "prompt_wav": str(prompt),
        "output_path": str(output),
    }
    stdout, stderr = proc.communicate(
        json.dumps(init) + "\n" + json.dumps(request) + "\n",
        timeout=10,
    )

    responses = [json.loads(line) for line in stdout.splitlines()]
    assert proc.returncode == 0, stderr
    assert responses[0]["status"] == "ready"
    assert responses[1]["status"] == "ok"
    assert output.read_bytes() == b"RIFF-fake-wave"
    assert "third-party progress" in stderr
