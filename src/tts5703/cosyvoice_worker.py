"""Isolated JSON-lines worker for CosyVoice3 inference.

This script is launched with CosyVoice's Python interpreter, not the main
project interpreter. Standard output is reserved exclusively for protocol
messages; third-party output is redirected to standard error.
"""

from __future__ import annotations

import json
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, TextIO

PROTOCOL_STDOUT = sys.stdout
END_OF_PROMPT = "<|endofprompt|>"


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


def _required_string(message: dict[str, Any], name: str) -> str:
    value = message.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_number(message: dict[str, Any], name: str, default: float) -> float:
    value = message.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _initialise(init: dict[str, Any]):
    repo_dir = Path(_required_string(init, "repo_dir")).resolve()
    model_dir = Path(_required_string(init, "model_dir")).resolve()
    matcha_dir = repo_dir / "third_party" / "Matcha-TTS"

    if not repo_dir.is_dir():
        raise FileNotFoundError(f"CosyVoice repository not found: {repo_dir}")
    if not matcha_dir.is_dir():
        raise FileNotFoundError(f"Matcha-TTS dependency not found: {matcha_dir}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"CosyVoice model directory not found: {model_dir}")
    if not (model_dir / "cosyvoice3.yaml").is_file():
        raise FileNotFoundError(
            f"CosyVoice3 configuration not found: {model_dir / 'cosyvoice3.yaml'}"
        )

    # CosyVoice's own examples require both paths on PYTHONPATH.
    sys.path.insert(0, str(matcha_dir))
    sys.path.insert(0, str(repo_dir))

    from cosyvoice.cli.cosyvoice import CosyVoice3

    return CosyVoice3(
        model_dir=str(model_dir),
        load_trt=bool(init.get("load_trt", False)),
        load_vllm=bool(init.get("load_vllm", False)),
        fp16=bool(init.get("fp16", False)),
    )


def _synthesise(model: Any, request: dict[str, Any]) -> dict[str, Any]:
    text = _required_string(request, "text")
    prompt_wav = Path(_required_string(request, "prompt_wav")).resolve()
    output_path = Path(_required_string(request, "output_path")).resolve()
    mode = request.get("mode", "zero_shot")
    if mode not in {"zero_shot", "instruct2"}:
        raise ValueError(f"unsupported CosyVoice mode: {mode!r}")
    speed = _positive_number(request, "speed", 1.0)

    if not prompt_wav.is_file():
        raise FileNotFoundError(f"CosyVoice prompt audio not found: {prompt_wav}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    import torchaudio

    if mode == "zero_shot":
        prompt_text = _required_string(request, "prompt_text")
        results = model.inference_zero_shot(
            text,
            prompt_text,
            str(prompt_wav),
            stream=False,
            speed=speed,
        )
    else:
        instruction = _required_string(request, "instruction")
        if instruction.count(END_OF_PROMPT) != 1 or not instruction.endswith(
            END_OF_PROMPT
        ):
            raise ValueError(
                "instruction must end with exactly one <|endofprompt|> marker"
            )
        results = model.inference_instruct2(
            text,
            instruction,
            str(prompt_wav),
            stream=False,
            speed=speed,
        )

    chunks = []
    for result in results:
        speech = result.get("tts_speech")
        if speech is None:
            raise RuntimeError("CosyVoice result is missing tts_speech")
        chunks.append(speech.detach().cpu())
    if not chunks:
        raise RuntimeError("CosyVoice generated no audio")

    audio = torch.cat(chunks, dim=1)
    torchaudio.save(
        str(output_path),
        audio,
        model.sample_rate,
        backend="soundfile",
    )
    return {
        "status": "ok",
        "output_path": str(output_path),
        "sample_rate": model.sample_rate,
        "samples": int(audio.shape[1]),
        "mode": mode,
        "speed": speed,
    }


def main() -> int:
    """Load one model, acknowledge readiness, then serve requests until stdin EOF.

    Initialisation failure terminates the worker. Per-request failures instead
    produce an error response and keep the loaded model alive, allowing the main
    process to decide whether a later dialogue should reuse it.
    """
    try:
        init = _read_message(sys.stdin)
        if init is None:
            raise RuntimeError("worker received no initialisation message")
        with redirect_stdout(sys.stderr):
            model = _initialise(init)
    except Exception as error:  # noqa: BLE001 - report failures across process boundary
        traceback.print_exc(file=sys.stderr)
        _respond({"status": "error", "message": f"{type(error).__name__}: {error}"})
        return 1

    _respond({"status": "ready", "sample_rate": model.sample_rate})
    while True:
        try:
            request = _read_message(sys.stdin)
            if request is None:
                return 0
            with redirect_stdout(sys.stderr):
                response = _synthesise(model, request)
        except Exception as error:  # noqa: BLE001 - keep worker usable for next request
            traceback.print_exc(file=sys.stderr)
            response = {
                "status": "error",
                "message": f"{type(error).__name__}: {error}",
            }
        _respond(response)


if __name__ == "__main__":
    raise SystemExit(main())
