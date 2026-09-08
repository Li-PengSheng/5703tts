"""Offline unit tests for provisional CosyVoice3 backend control mappings."""

import asyncio
from pathlib import Path

import pytest

from tts5703 import tts_engine
from tts5703.cosyvoice_controls import (
    BackendControlError,
    build_cosyvoice_instruction,
    rate_to_cosyvoice_speed,
)
from tts5703.tts_engine import build_cosyvoice_request
from tts5703.validate import NormalizedTurn

END_OF_PROMPT = "<|endofprompt|>"
PREFIX = "You are a helpful assistant."
AFFECT_CLAUSES = {
    "neutral": "Speak naturally in a neutral, emotionally even conversational manner.",
    "sad": "Speak in a clearly sad and downcast manner.",
    "anxious": (
        "Speak in a clearly anxious, worried, and uneasy manner, with noticeable "
        "nervous tension and uncertainty, but not panic or urgency."
    ),
    "angry": "Speak in a clearly angry, firm, and forceful manner, without shouting.",
    "warm": (
        "Speak in a clearly warm, gentle, compassionate, supportive, and reassuring "
        "manner, with an emotionally present and caring delivery rather than a "
        "cheerful or excited one."
    ),
    "distressed": "Use a distressed, worried, and sad tone.",
}
AROUSAL_CLAUSES = {
    "low": (
        "Keep vocal activation clearly low, with subdued energy and restrained "
        "emphasis. Do not deliberately change the speaking rate."
    ),
    "medium": None,
    "high": (
        "Use clearly high vocal activation, with stronger energy, projection, and "
        "emphasis, and a more animated delivery. Do not deliberately change the "
        "speaking rate."
    ),
}


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


def _expected_instruction(arousal: str | None, coarse_affect: str | None) -> str | None:
    if arousal is None and coarse_affect is None:
        return None
    clauses = [AFFECT_CLAUSES.get(coarse_affect), AROUSAL_CLAUSES.get(arousal)]
    return f"{PREFIX} {' '.join(clause for clause in clauses if clause)}{END_OF_PROMPT}"


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
    assert rate_to_cosyvoice_speed(rate) == speed


def test_null_controls_produce_no_instruction() -> None:
    assert build_cosyvoice_instruction(None, None) is None


@pytest.mark.parametrize(
    ("arousal", "expected_control"),
    [
        ("low", AROUSAL_CLAUSES["low"]),
        ("medium", None),
        ("high", AROUSAL_CLAUSES["high"]),
    ],
)
def test_arousal_instruction_is_deterministic(
    arousal: str, expected_control: str | None
) -> None:
    expected_body = f" {expected_control}" if expected_control else " "
    assert build_cosyvoice_instruction(arousal, None) == (
        f"{PREFIX}{expected_body}{END_OF_PROMPT}"
    )


def test_low_and_high_arousal_explicitly_preserve_speaking_rate() -> None:
    for arousal in ("low", "high"):
        assert AROUSAL_CLAUSES[arousal] is not None
        assert (
            "Do not deliberately change the speaking rate." in AROUSAL_CLAUSES[arousal]
        )


@pytest.mark.parametrize(
    ("coarse_affect", "expected_control"),
    list(AFFECT_CLAUSES.items()),
)
def test_coarse_affect_instruction_is_deterministic(
    coarse_affect: str, expected_control: str
) -> None:
    assert build_cosyvoice_instruction(None, coarse_affect) == (
        f"{PREFIX} {expected_control}{END_OF_PROMPT}"
    )


@pytest.mark.parametrize(
    ("arousal", "coarse_affect"),
    [
        ("medium", "neutral"),
        ("medium", "sad"),
        ("low", "anxious"),
        ("high", "angry"),
        ("low", "warm"),
        ("high", "distressed"),
    ],
)
def test_affect_then_arousal_composition(arousal: str, coarse_affect: str) -> None:
    instruction = build_cosyvoice_instruction(arousal, coarse_affect)

    assert instruction == _expected_instruction(arousal, coarse_affect)
    assert instruction is not None
    assert instruction.count(END_OF_PROMPT) == 1
    assert instruction.endswith(END_OF_PROMPT)


@pytest.mark.parametrize(
    ("arousal", "coarse_affect", "field"),
    [
        ("frantic", None, "arousal"),
        (None, "cheerful", "coarse_affect"),
    ],
)
def test_unknown_controls_still_fail(
    arousal: str | None, coarse_affect: str | None, field: str
) -> None:
    with pytest.raises(BackendControlError, match=f"CosyVoice {field} mapping"):
        build_cosyvoice_instruction(arousal, coarse_affect)


@pytest.mark.parametrize(
    ("arousal", "coarse_affect"),
    [
        ("low", None),
        ("medium", None),
        ("high", None),
        (None, "neutral"),
        ("medium", "sad"),
        ("low", "anxious"),
        ("high", "angry"),
        ("low", "warm"),
        ("high", "distressed"),
    ],
)
def test_generated_instruction_has_exactly_one_end_marker(
    arousal: str | None, coarse_affect: str | None
) -> None:
    instruction = build_cosyvoice_instruction(arousal, coarse_affect)
    assert instruction is not None
    assert instruction.count(END_OF_PROMPT) == 1
    assert instruction.endswith(END_OF_PROMPT)


@pytest.mark.parametrize(
    ("rate", "arousal", "coarse_affect", "mode", "speed"),
    [
        ("normal", None, None, "zero_shot", 1.0),
        ("slow", None, None, "zero_shot", 0.8),
        ("fast", None, None, "zero_shot", 1.2),
        ("normal", "low", None, "instruct2", 1.0),
        ("normal", "medium", None, "instruct2", 1.0),
        ("normal", "high", None, "instruct2", 1.0),
        ("normal", "medium", "sad", "instruct2", 1.0),
        ("normal", "low", "anxious", "instruct2", 1.0),
        ("normal", "high", "angry", "instruct2", 1.0),
        ("normal", "low", "warm", "instruct2", 1.0),
        ("normal", "high", "distressed", "instruct2", 1.0),
    ],
)
def test_request_routing_and_mapping(
    rate: str,
    arousal: str | None,
    coarse_affect: str | None,
    mode: str,
    speed: float,
) -> None:
    request = _request(_turn(rate=rate, arousal=arousal, coarse_affect=coarse_affect))
    expected_instruction = _expected_instruction(arousal, coarse_affect)

    assert request["mode"] == mode
    assert request["speed"] == speed
    if expected_instruction is None:
        assert "instruction" not in request
    else:
        assert request["instruction"] == expected_instruction


@pytest.mark.parametrize(("rate", "speed"), [("normal", 1.0), ("slow", 0.8)])
@pytest.mark.parametrize("arousal", ["low", "high"])
def test_arousal_does_not_change_numeric_speed(
    rate: str, speed: float, arousal: str
) -> None:
    assert _request(_turn(rate=rate, arousal=arousal))["speed"] == speed


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
            "You are a helpful assistant. Use a distressed, worried, and sad tone. "
            "Use clearly high vocal activation, with stronger energy, projection, "
            "and emphasis, and a more animated delivery. Do not deliberately change "
            "the speaking rate.<|endofprompt|>"
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
    assert "high vocal activation" in captured_request["instruction"]
    assert turn.text == captured_request["text"]
