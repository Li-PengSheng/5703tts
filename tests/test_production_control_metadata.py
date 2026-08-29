"""Requested-versus-supported control reporting in production metadata."""

import copy
from pathlib import Path

import pytest

from tts5703.assemble import TurnTiming
from tts5703.config import load_config
from tts5703.metadata import build_metadata
from tts5703.tts_engine import describe_engine

CONFIG_PATH = Path("config/config.yaml")
KOKORO_CONFIG_PATH = Path("config/config.kokoro.yaml")


def _timing(
    *,
    arousal: str | None = None,
    coarse_affect: str | None = None,
    emotion: str | None = None,
    paralinguistic_events: list | None = None,
    pause_before_ms: int = 150,
    pause_after_ms: int = 700,
) -> TurnTiming:
    return TurnTiming(
        turn_id=1,
        speaker="caller",
        text="Please stay with me.",
        label="alert",
        rate="slow",
        pause_before_ms=pause_before_ms,
        pause_after_ms=pause_after_ms,
        emotion=emotion,
        arousal=arousal,
        coarse_affect=coarse_affect,
        paralinguistic_events=paralinguistic_events
        if paralinguistic_events is not None
        else [],
        start_sec=0.15,
        end_sec=0.65,
    )


def _metadata(timing: TurnTiming, engine_info: dict) -> dict:
    return build_metadata(
        "dialogue001",
        Path("dialogue001_clean.wav"),
        Path("dialogue001_telephone.wav"),
        [timing],
        {1: Path("turn_001.wav")},
        engine_info,
    )


def _turn_metadata(timing: TurnTiming, config_path: Path) -> dict:
    engine_info = describe_engine(load_config(config_path))
    return _metadata(timing, engine_info)["turns"][0]


def test_kokoro_marks_requested_arousal_as_ignored_but_preserves_it() -> None:
    turn = _turn_metadata(_timing(arousal="high"), KOKORO_CONFIG_PATH)

    assert turn["arousal"] == "high"
    assert turn["requested_acoustic_spec"]["arousal"] == "high"
    assert turn["ignored_requested_controls"] == ["arousal"]


def test_kokoro_marks_requested_coarse_affect_as_ignored_but_preserves_it() -> None:
    turn = _turn_metadata(_timing(coarse_affect="distressed"), KOKORO_CONFIG_PATH)

    assert turn["coarse_affect"] == "distressed"
    assert turn["requested_acoustic_spec"]["coarse_affect"] == "distressed"
    assert turn["ignored_requested_controls"] == ["coarse_affect"]


def test_kokoro_control_support_marks_unsupported_controls() -> None:
    metadata = _metadata(
        _timing(arousal="high"), describe_engine(load_config(KOKORO_CONFIG_PATH))
    )

    assert metadata["tts"]["control_support"] == {
        "rate": "model_control",
        "pause_before_ms": "pipeline_timing",
        "pause_after_ms": "pipeline_timing",
        "arousal": "unsupported",
        "coarse_affect": "unsupported",
        "emotion": "unsupported",
        "paralinguistic_events": "unsupported",
    }


def test_null_emotion_is_not_reported_as_ignored() -> None:
    turn = _turn_metadata(_timing(emotion=None), KOKORO_CONFIG_PATH)

    assert turn["emotion"] is None
    assert turn["requested_acoustic_spec"]["emotion"] is None
    assert "emotion" not in turn["ignored_requested_controls"]


def test_requested_pauses_are_not_reported_as_ignored() -> None:
    turn = _turn_metadata(
        _timing(pause_before_ms=800, pause_after_ms=1000), KOKORO_CONFIG_PATH
    )

    assert turn["requested_acoustic_spec"]["pause_before_ms"] == 800
    assert turn["requested_acoustic_spec"]["pause_after_ms"] == 1000
    assert turn["ignored_requested_controls"] == []


def test_cosyvoice_arousal_and_coarse_affect_are_not_reported_as_ignored() -> None:
    timing = _timing(arousal="high", coarse_affect="distressed")
    metadata = _metadata(timing, describe_engine(load_config(CONFIG_PATH)))
    turn = metadata["turns"][0]

    assert turn["ignored_requested_controls"] == []
    assert metadata["tts"]["control_support"]["arousal"] == "provisional_model_control"
    assert (
        metadata["tts"]["control_support"]["coarse_affect"]
        == "provisional_model_control"
    )
    assert metadata["tts"]["control_mapping"] == "provisional"


def test_requested_acoustic_spec_preserves_the_full_requested_intent() -> None:
    timing = _timing(
        arousal="high",
        coarse_affect="distressed",
        emotion="fearful",
        paralinguistic_events=["sigh", {"type": "sob"}],
    )
    turn = _turn_metadata(timing, CONFIG_PATH)

    assert turn["requested_acoustic_spec"] == {
        "rate": "slow",
        "pause_before_ms": 150,
        "pause_after_ms": 700,
        "arousal": "high",
        "coarse_affect": "distressed",
        "emotion": "fearful",
        "paralinguistic_events": ["sigh", {"type": "sob"}],
    }
    assert turn["ignored_requested_controls"] == [
        "emotion",
        "paralinguistic_events",
    ]


def test_flat_legacy_fields_still_mirror_the_requested_spec() -> None:
    timing = _timing(arousal="high", coarse_affect="distressed", emotion="fearful")
    turn = _turn_metadata(timing, CONFIG_PATH)

    for field, value in turn["requested_acoustic_spec"].items():
        assert turn[field] == value


def test_engine_without_declared_capabilities_reports_null_not_empty() -> None:
    metadata = _metadata(_timing(arousal="high"), {"engine": "edge_tts"})

    assert metadata["tts"]["control_support"] is None
    assert metadata["turns"][0]["ignored_requested_controls"] is None
    assert metadata["turns"][0]["requested_acoustic_spec"]["arousal"] == "high"


def test_cosyvoice_reproducibility_config_appears_in_metadata() -> None:
    config = load_config(CONFIG_PATH)
    metadata = _metadata(_timing(), describe_engine(config))
    cosyvoice = config["tts"]["cosyvoice"]
    tts_metadata = metadata["tts"]

    assert tts_metadata["engine"] == "cosyvoice"
    assert tts_metadata["model"] == "Fun-CosyVoice3-0.5B"
    assert tts_metadata["model_dir"] == cosyvoice["model_dir"]
    assert tts_metadata["repo_dir"] == cosyvoice["repo_dir"]
    assert tts_metadata["fp16"] == cosyvoice["fp16"]
    assert tts_metadata["load_trt"] == cosyvoice["load_trt"]
    assert tts_metadata["load_vllm"] == cosyvoice["load_vllm"]
    assert set(tts_metadata["prompts"]) == set(cosyvoice["voice_map"])
    for speaker, prompt in tts_metadata["prompts"].items():
        assert prompt["prompt_wav"] == cosyvoice["voice_map"][speaker]["prompt_wav"]
        assert prompt["prompt_text"] == cosyvoice["voice_map"][speaker]["prompt_text"]


def test_cosyvoice_expected_sample_rate_defaults_to_the_model_rate() -> None:
    engine_info = describe_engine(load_config(CONFIG_PATH))

    assert engine_info["expected_sample_rate"] == 24_000
    assert engine_info["expected_sample_rate_source"] == "model_default"


def test_cosyvoice_runtime_sample_rate_stays_null_when_unavailable() -> None:
    engine_info = describe_engine(load_config(CONFIG_PATH))

    assert engine_info["runtime_sample_rate"] is None
    assert engine_info["sample_rate_verification"] == "not_runtime_verified"


def test_configured_cosyvoice_sample_rate_is_expected_not_verified() -> None:
    config = copy.deepcopy(load_config(CONFIG_PATH))
    config["tts"]["cosyvoice"]["sample_rate"] = 16_000

    engine_info = describe_engine(config)

    assert engine_info["expected_sample_rate"] == 16_000
    assert engine_info["expected_sample_rate_source"] == "config"
    assert engine_info["runtime_sample_rate"] is None
    assert engine_info["sample_rate_verification"] == "not_runtime_verified"


def test_cosyvoice_metadata_never_claims_an_actual_output_sample_rate() -> None:
    config = copy.deepcopy(load_config(CONFIG_PATH))
    config["tts"]["cosyvoice"]["sample_rate"] = 16_000
    metadata = _metadata(_timing(), describe_engine(config))
    tts_metadata = metadata["tts"]

    # A bare "sample_rate", or anything named after the output, would read as an
    # observed property of the written audio, which nothing here has verified.
    misleading_keys = [
        key for key in tts_metadata if key == "sample_rate" or "output_sample" in key
    ]
    assert misleading_keys == []
    assert tts_metadata["expected_sample_rate"] == 16_000


def test_kokoro_engine_info_still_records_its_sample_rate() -> None:
    engine_info = describe_engine(load_config(KOKORO_CONFIG_PATH))

    assert engine_info["sample_rate"] == 24_000
    assert engine_info["voices"] == {"counsellor": "af_heart", "caller": "am_adam"}


@pytest.mark.parametrize(
    "field",
    ["turn_id", "speaker", "text", "label", "turn_audio", "start_time", "end_time"],
)
def test_existing_metadata_contract_fields_are_preserved(field: str) -> None:
    turn = _turn_metadata(_timing(), CONFIG_PATH)

    assert field in turn
