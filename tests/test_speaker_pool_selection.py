"""Offline checks for the VCTK v0.1 speaker-pool shortlist after Week-5 QA.

These tests read committed JSON only. They do not load CosyVoice, a GPU, or
the network. ``resolve_selection`` is the same mapping the materialiser writes
into ``speaker_registry.json``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = ROOT / "data" / "speaker_pool" / "vctk_v0.1" / "selected_speakers.json"
CANDIDATE_REPORT_PATH = (
    ROOT / "data" / "speaker_pool" / "vctk_v0.1" / "candidate_report.json"
)
REGISTRY_PATH = ROOT / "data" / "speaker_pool" / "vctk_v0.1" / "speaker_registry.json"
MATERIALIZE_PATH = ROOT / "scripts" / "materialize_vctk_speaker_pool.py"
MATERIALIZE_SPEC = importlib.util.spec_from_file_location(
    "vctk_materialize_selection", MATERIALIZE_PATH
)
assert MATERIALIZE_SPEC is not None and MATERIALIZE_SPEC.loader is not None
materialize = importlib.util.module_from_spec(MATERIALIZE_SPEC)
MATERIALIZE_SPEC.loader.exec_module(materialize)

EXPECTED_REFERENCES = {
    "spk_001": ("p248_112", "p248_064"),
    "spk_002": ("p253_262", "p253_147"),
    "spk_003": ("p261_199", "p261_072"),
    "spk_004": ("p234_238", "p234_030"),
    "spk_005": ("p239_004", "p239_006"),
    "spk_006": ("p283_013", "p283_020"),
    "spk_007": ("p251_009", "p251_364"),
    "spk_008": ("p245_269", "p245_309"),
    "spk_009": ("p237_013", "p237_291"),
    "spk_010": ("p243_394", "p243_349"),
    "spk_011": ("p256_241", "p256_223"),
    "spk_012": ("p252_007", "p252_385"),
}
QA_NOTE_SPEAKERS = ("spk_002", "spk_005", "spk_008", "spk_009", "spk_010")


def _selection() -> dict:
    return json.loads(SELECTION_PATH.read_text(encoding="utf-8"))


def _speakers() -> dict[str, dict]:
    return {entry["speaker_id"]: entry for entry in _selection()["speakers"]}


def test_spk_008_primary_is_the_promoted_backup() -> None:
    speaker = _speakers()["spk_008"]
    assert speaker["source_speaker_id"] == "p245"
    assert speaker["primary_utterance_id"] == "p245_269"
    assert speaker["backup_utterance_id"] == "p245_309"


def test_spk_009_primary_is_the_promoted_backup() -> None:
    speaker = _speakers()["spk_009"]
    assert speaker["source_speaker_id"] == "p237"
    assert speaker["primary_utterance_id"] == "p237_013"
    assert speaker["backup_utterance_id"] == "p237_291"


def test_spk_010_primary_is_the_promoted_backup() -> None:
    speaker = _speakers()["spk_010"]
    assert speaker["source_speaker_id"] == "p243"
    assert speaker["primary_utterance_id"] == "p243_394"
    assert speaker["backup_utterance_id"] == "p243_349"


def test_spk_002_backup_is_the_final_alternate() -> None:
    speaker = _speakers()["spk_002"]
    assert speaker["source_speaker_id"] == "p253"
    assert speaker["primary_utterance_id"] == "p253_262"
    assert speaker["backup_utterance_id"] == "p253_147"


def test_spk_005_backup_is_the_final_alternate() -> None:
    speaker = _speakers()["spk_005"]
    assert speaker["source_speaker_id"] == "p239"
    assert speaker["primary_utterance_id"] == "p239_004"
    assert speaker["backup_utterance_id"] == "p239_006"


def test_spk_002_and_spk_005_are_not_auto_promoted() -> None:
    speakers = _speakers()
    assert speakers["spk_002"]["primary_utterance_id"] == "p253_262"
    assert speakers["spk_005"]["primary_utterance_id"] == "p239_004"
    for speaker_id in ("spk_002", "spk_005"):
        note = speakers[speaker_id]["human_selection_note"].lower()
        assert "auto-promoted" in note or "pending human listening" in note


def test_unreviewed_speakers_are_unchanged() -> None:
    speakers = _speakers()
    assert list(speakers) == list(EXPECTED_REFERENCES)
    for speaker_id, (primary, backup) in EXPECTED_REFERENCES.items():
        assert speakers[speaker_id]["primary_utterance_id"] == primary
        assert speakers[speaker_id]["backup_utterance_id"] == backup


def test_selection_method_records_manual_smoke_review_not_automatic_scoring() -> None:
    selection = _selection()
    assert selection["selection_status"] == "provisional_smoke_test_shortlist"
    method = selection["selection_method"]
    assert "ranked VCTK candidates" in method
    assert "manual CosyVoice3" in method
    assert "smoke-test review" in method
    lowered = method.lower()
    assert "automatic" not in lowered
    assert "score" not in lowered
    assert "rank 1 used as primary" not in lowered


def test_affected_speakers_carry_a_short_human_selection_note() -> None:
    speakers = _speakers()
    for speaker_id in QA_NOTE_SPEAKERS:
        note = speakers[speaker_id].get("human_selection_note")
        assert isinstance(note, str) and note.strip()
    for speaker_id, entry in speakers.items():
        if speaker_id not in QA_NOTE_SPEAKERS:
            assert "human_selection_note" not in entry


def test_resolved_selection_matches_the_week5_qa_utterances() -> None:
    report = materialize.load_candidate_report(CANDIDATE_REPORT_PATH)
    selection = materialize.load_selection(SELECTION_PATH)
    resolved = materialize.resolve_selection(report, selection)
    assert [entry["speaker_id"] for entry in resolved] == list(EXPECTED_REFERENCES)
    for entry in resolved:
        primary, backup = EXPECTED_REFERENCES[entry["speaker_id"]]
        assert entry["references"]["primary"]["source_utterance_id"] == primary
        assert entry["references"]["backup"]["source_utterance_id"] == backup


def test_committed_registry_matches_selected_speakers() -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    speakers = {entry["speaker_id"]: entry for entry in registry["speakers"]}
    assert list(speakers) == list(EXPECTED_REFERENCES)
    for speaker_id, (primary, backup) in EXPECTED_REFERENCES.items():
        entry = speakers[speaker_id]
        assert entry["primary_reference"]["source_utterance_id"] == primary
        assert entry["backup_reference"]["source_utterance_id"] == backup
        assert entry["primary_reference"]["prompt_wav"] == (
            f"data/speaker_pool/vctk_v0.1/references/{speaker_id}_primary.wav"
        )
        assert entry["backup_reference"]["prompt_wav"] == (
            f"data/speaker_pool/vctk_v0.1/references/{speaker_id}_backup.wav"
        )
    assert registry["status"] == "provisional_smoke_test"
    method = registry["selection"]["selection_method"]
    assert "manual CosyVoice3" in method
    assert registry["selection"]["selection_status"] == (
        "provisional_smoke_test_shortlist"
    )
