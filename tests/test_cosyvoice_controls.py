"""Offline unit tests for provisional CosyVoice3 backend control mappings."""

import asyncio
from pathlib import Path

import pytest

from tts5703 import tts_engine
from tts5703.tts_engine import (
    build_cosyvoice_instruction,
    build_cosyvoice_request,
    rate_to_cosyvoice_speed,
)
from tts5703.validate import NormalizedTurn

END_OF_PROMPT = "<|endofprompt|>"


def _turn(
    *,
    rate: str = "normal",
    arousal: str | None = None,
    coarse_affect: str | None = None,
) -> NormalizedTurn:
    return NormalizedTurn(
        turn_id=7,
        speaker="caller",
        text="Please stay with me while we decide what to do next.",
        label="alert",
        rate=rate,
        pause_before_ms=800,
        pause_after_ms=1000,
        arousal=arousal,
        coarse_affect=coarse_affect,
        emotion="fearful",
        paralinguistic_events=["sigh"],
    )


def _request(turn: NormalizedTurn) -> dict:
    return build_cosyvoice_request(
        turn,
        "Reference transcript.",
        Path("prompt.wav"),
        Path("output.wav"),
    )


@pytest.mark.parametrize(
    ("rate", "speed"),
    [
        ("slow", 0.8),
        ("normal", 1.0),
        ("fast", 1.2),
        ("-20%", 0.8),
        ("+20%", 1.2),
    ],
)
def test_rate_maps_to_cosyvoice_speed(rate: str, speed: float) -> None:
    assert rate_to_cosyvoice_speed(rate) == pytest.approx(speed)


def test_null_controls_produce_no_instruction() -> None:
    assert build_cosyvoice_instruction(None, None) is None


@pytest.mark.parametrize(
    ("arousal", "expected_control"),
    [
        ("low", "Use a calm, soft, subdued delivery."),
        ("medium", "Use a neutral, moderately expressive delivery."),
        ("high", "Use an energetic, intense delivery."),
    ],
)
def test_arousal_instruction_is_deterministic(
    arousal: str, expected_control: str
) -> None:
    assert build_cosyvoice_instruction(arousal, None) == (
        f"You are a helpful assistant. {expected_control}{END_OF_PROMPT}"
    )


@pytest.mark.parametrize(
    ("coarse_affect", "expected_control"),
    [
        ("neutral", "Use a neutral, composed tone."),
        ("distressed", "Use a distressed, worried, and sad tone."),
    ],
)
def test_coarse_affect_instruction_is_deterministic(
    coarse_affect: str, expected_control: str
) -> None:
    assert build_cosyvoice_instruction(None, coarse_affect) == (
        f"You are a helpful assistant. {expected_control}{END_OF_PROMPT}"
    )


def test_combined_instruction_is_deterministic() -> None:
    assert build_cosyvoice_instruction("high", "distressed") == (
        "You are a helpful assistant. Use an energetic, intense delivery. "
        "Use a distressed, worried, and sad tone.<|endofprompt|>"
    )


@pytest.mark.parametrize(
    ("arousal", "coarse_affect"),
    [
        ("low", None),
        ("high", None),
        (None, "neutral"),
        (None, "distressed"),
        ("medium", "distressed"),
    ],
)
def test_generated_instruction_has_exactly_one_end_marker(
    arousal: str | None, coarse_affect: str | None
) -> None:
    instruction = build_cosyvoice_instruction(arousal, coarse_affect)
    assert instruction is not None
    assert instruction.count(END_OF_PROMPT) == 1
    assert instruction.endswith(END_OF_PROMPT)


def test_rate_only_request_uses_zero_shot() -> None:
    request = _request(_turn(rate="slow"))
    assert request["mode"] == "zero_shot"
    assert request["speed"] == pytest.approx(0.8)
    assert "instruction" not in request


def test_explicit_arousal_request_uses_instruct2() -> None:
    request = _request(_turn(arousal="high"))
    assert request["mode"] == "instruct2"
    assert "energetic, intense" in request["instruction"]


def test_explicit_coarse_affect_request_uses_instruct2() -> None:
    request = _request(_turn(coarse_affect="neutral"))
    assert request["mode"] == "instruct2"
    assert "neutral, composed" in request["instruction"]


def test_request_transforms_controls_only_at_cosyvoice_boundary() -> None:
    turn = _turn(rate="fast", arousal="high", coarse_affect="distressed")
    request = _request(turn)

    assert request == {
        "text": turn.text,
        "prompt_text": "Reference transcript.",
        "prompt_wav": "prompt.wav",
        "output_path": "output.wav",
        "speed": 1.2,
        "mode": "instruct2",
        "instruction": (
            "You are a helpful assistant. Use an energetic, intense delivery. "
            "Use a distressed, worried, and sad tone.<|endofprompt|>"
        ),
    }
    assert turn.text not in request["instruction"]
    assert turn.label not in request["instruction"]
    assert not {
        "rate",
        "arousal",
        "coarse_affect",
        "pause_before_ms",
        "pause_after_ms",
        "emotion",
        "paralinguistic_events",
    }.intersection(request)


def test_cosyvoice_engine_sends_transformed_request_without_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_dir = tmp_path / "CosyVoice"
    model_dir = tmp_path / "model"
    repo_dir.mkdir()
    model_dir.mkdir()
    python_bin = tmp_path / "python"
    prompt_wav = tmp_path / "prompt.wav"
    python_bin.write_text("fake", encoding="utf-8")
    prompt_wav.write_bytes(b"RIFF-fake-prompt")
    config = {
        "tts": {
            "engine": "cosyvoice",
            "cosyvoice": {
                "python_bin": str(python_bin),
                "repo_dir": str(repo_dir),
                "model_dir": str(model_dir),
                "load_trt": False,
                "load_vllm": False,
                "fp16": False,
                "voice_map": {
                    "caller": {
                        "prompt_wav": str(prompt_wav),
                        "prompt_text": "Reference transcript.",
                    }
                },
            },
        }
    }
    captured_request: dict = {}

    monkeypatch.setattr(tts_engine, "_get_cosyvoice_worker", lambda *args: object())

    def fake_worker_request(worker: object, request: dict) -> dict:
        captured_request.update(request)
        Path(request["output_path"]).write_bytes(b"RIFF-fake-output")
        return {"status": "ok"}

    monkeypatch.setattr(tts_engine, "_cosyvoice_request", fake_worker_request)
    turn = _turn(rate="slow", arousal="high", coarse_affect=None)

    output_path = asyncio.run(tts_engine.synthesize_turn(turn, tmp_path, config))

    assert output_path.read_bytes() == b"RIFF-fake-output"
    assert captured_request["mode"] == "instruct2"
    assert captured_request["speed"] == pytest.approx(0.8)
    assert "energetic, intense" in captured_request["instruction"]
    assert turn.text == captured_request["text"]
