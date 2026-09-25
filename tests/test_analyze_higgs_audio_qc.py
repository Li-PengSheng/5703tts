"""Small synthetic checks for the offline Higgs audio diagnostic."""

import importlib.util
import json
import math
import struct
import sys
import wave
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_higgs_audio_qc.py"
spec = importlib.util.spec_from_file_location("analyze_higgs_audio_qc", SCRIPT)
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)
analyze, audio_metrics = analyzer.analyze, analyzer.audio_metrics
json_safe, report = analyzer.json_safe, analyzer.report


def wav(
    path: Path, sections: list[tuple[float, int]], *, rate=8000, channels=1
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        for seconds, amplitude in sections:
            frames = round(seconds * rate)
            samples = [amplitude * (-1 if index % 2 else 1) for index in range(frames)]
            output.writeframes(
                struct.pack(
                    f"<{frames * channels}h",
                    *(sample for sample in samples for _ in range(channels)),
                )
            )


def metadata(folder: Path, dialogue_id: str, turns: list[dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{dialogue_id}_metadata.json").write_text(
        json.dumps({"dialogue_id": dialogue_id, "turns": turns}), encoding="utf-8"
    )


def turn(ordinal: int, text: str = "a few words") -> dict:
    return {
        "source_identity": {"ordinal": ordinal, "source_turn_id": ordinal - 1},
        "execution": {"turn_audio": f"turn_{ordinal:03d}.wav"},
        "planned": {"higgs": {"text": text, "model_input": f"<|emotion:calm|>{text}"}},
    }


def test_relative_tail_detects_residue_above_minus_40_db(tmp_path: Path) -> None:
    path = tmp_path / "tail.wav"
    wav(path, [(3, 10000), (12, 500)], channels=2)
    result = audio_metrics(path)
    assert result["duration_sec"] == 15
    assert result["tail_dbfs"]["5"] > -40
    assert result["tail_5_drop_db"] == pytest.approx(26.02, abs=0.1)
    assert result["relative_thresholds"]["18"]["trailing_low_sec"] == 12
    assert result["relative_thresholds"]["18"]["low_fraction_last_5_sec"] == 1
    assert result["relative_thresholds"]["18"]["low_fraction_last_10_sec"] == 1


def test_short_speech_and_true_silence(tmp_path: Path) -> None:
    short = tmp_path / "short.wav"
    wav(short, [(0.4, 8000)], rate=11025)
    result = audio_metrics(short)
    assert result["duration_sec"] == pytest.approx(0.4, abs=0.0001)
    assert result["tail_dbfs"]["10"] is None
    assert result["relative_thresholds"]["6"]["low_fraction_last_10_sec"] is None
    assert result["relative_thresholds"]["6"]["trailing_low_sec"] == 0

    silence = tmp_path / "silence.wav"
    wav(silence, [(1, 10000), (5, 0)])
    result = audio_metrics(silence)
    assert result["tail_dbfs"]["2"] == -math.inf
    assert result["tail_5_drop_db"] == math.inf
    assert result["relative_thresholds"]["18"]["trailing_low_sec"] == 5


def test_all_silence_and_empty_text_are_serializable(tmp_path: Path) -> None:
    folder = tmp_path / "d"
    wav(folder / "turn_001.wav", [(0.5, 0)])
    metadata(folder, "d", [turn(1, "")])
    rows, warnings = analyze(tmp_path)
    assert not warnings
    row = rows[0]
    assert row["sec_per_word"] is None and row["sec_per_char"] is None
    assert row["whole_dbfs"] == -math.inf
    assert row["body_dbfs"] == -math.inf
    assert row["tail_5_drop_db"] is None
    assert row["relative_thresholds"]["10"]["trailing_low_sec"] is None
    assert (
        json.loads(json.dumps(json_safe(row), allow_nan=False))["whole_dbfs"] == "-inf"
    )


def test_missing_and_malformed_metadata_are_reported_in_order(
    tmp_path: Path, capsys
) -> None:
    a, b, c = (tmp_path / name for name in ("a", "b", "c"))
    wav(a / "turn_002.wav", [(0.25, 5000)])
    wav(a / "turn_001.wav", [(0.25, 5000)])
    metadata(a, "a", [turn(2), turn(1)])
    metadata(b, "b", [turn(1)])  # Missing WAV.
    c.mkdir()
    (c / "c_metadata.json").write_text("{invalid", encoding="utf-8")
    wav(c / "turn_003.wav", [(0.25, 5000)])  # Orphan WAV.
    rows, warnings = analyze(tmp_path)
    assert [row["wav_path"] for row in rows] == ["a/turn_001.wav", "a/turn_002.wav"]
    assert len(warnings) == 3
    assert "b/" in warnings[0] and "c/c_metadata.json" in warnings[1]
    report(rows, warnings)
    first = capsys.readouterr().out
    report(rows, warnings)
    assert capsys.readouterr().out == first


def test_cli_writes_strict_json_without_overwriting_input(
    tmp_path: Path, monkeypatch
) -> None:
    folder = tmp_path / "d"
    wav(folder / "turn_001.wav", [(0.5, 0)])
    metadata(folder, "d", [turn(1)])
    output = tmp_path / "results.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--output-root", str(tmp_path), "--json-out", str(output)],
    )
    analyzer.main()
    assert (
        json.loads(output.read_text(encoding="utf-8"))["turns"][0]["whole_dbfs"]
        == "-inf"
    )
    with pytest.raises(FileExistsError):
        analyzer.main()
    assert (
        json.loads((folder / "d_metadata.json").read_text(encoding="utf-8"))[
            "dialogue_id"
        ]
        == "d"
    )
