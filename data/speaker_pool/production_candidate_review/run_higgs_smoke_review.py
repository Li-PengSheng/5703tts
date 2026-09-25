"""Review-only, same-text Higgs smoke for five human-selected VCTK references."""

import hashlib
import json
import time
from pathlib import Path

import yaml

from tts5703.backends import higgs

ROOT = Path(__file__).resolve().parents[3]
PACK = Path(__file__).resolve().parent
SELECTIONS = {
    "p227": ("B", "p227_020"),
    "p294": ("B", "p294_017"),
    "p305": ("B", "p305_114"),
    "p318": ("A", "p318_051"),
    "p343": ("A", "p343_012"),
}
TEXT = "I understand. Take your time and tell me what happened when you are ready."


def main() -> None:
    manifest = json.loads((PACK / "candidates.json").read_text(encoding="utf-8"))
    selected = [
        entry
        for entry in manifest["supplemental"]
        if entry.get("second_pass_status") == "source_listening_selected"
    ]
    if (
        len(selected) != len(SELECTIONS)
        or {entry["source_speaker_id"] for entry in selected} != SELECTIONS.keys()
    ):
        raise ValueError(
            "Human-selected sources must be exactly the five reviewed sources"
        )

    references = []
    for source, (choice, utterance) in SELECTIONS.items():
        entry = next(item for item in selected if item["source_speaker_id"] == source)
        if (
            entry.get("selected_reference_choice") != choice
            or entry.get("selected_source_utterance_id") != utterance
        ):
            raise ValueError(f"Human selection changed for {source}")
        candidate = next(
            (item for item in entry["candidates"] if item["candidate"] == choice),
            None,
        )
        if candidate is None or candidate.get("source_utterance_id") != utterance:
            raise ValueError(f"Selected utterance changed for {source}")
        if candidate.get("listening_wav") != f"wav/{source}_{choice.lower()}.wav":
            raise ValueError(f"Selected listening WAV changed for {source}")
        wav = PACK / candidate["listening_wav"]
        if not wav.is_file():
            raise FileNotFoundError(f"Selected listening WAV missing: {wav}")
        with wav.open("rb") as handle:
            actual_sha = hashlib.file_digest(handle, "sha256").hexdigest()
        expected_sha = candidate["listening_wav_sha256"]
        if actual_sha != expected_sha:
            raise ValueError(f"Selected listening WAV SHA-256 mismatch: {wav}")
        references.append((source, utterance, wav, actual_sha))

    config = yaml.safe_load(
        (ROOT / "config/config_higgs_cloud.yaml").read_text(encoding="utf-8")
    )["tts"]["higgs"]
    output_dir = PACK / "higgs_smoke_review"
    output_dir.mkdir(exist_ok=False)

    total_started = time.perf_counter()
    try:
        worker = higgs._get_worker(
            str((ROOT / config["server_executable"]).resolve()),
            str((ROOT / config["model_dir"]).resolve()),
            config["host"],
            config["port"],
            config["startup_timeout_seconds"],
            config["inference_timeout_seconds"],
        )
        for source, utterance, wav, reference_sha in references:
            output = output_dir / f"{source}_{utterance}.wav"
            started = time.perf_counter()
            response = higgs._request(
                worker,
                {
                    "model_input": TEXT,
                    "reference_wav": str(wav),
                    "output_path": str(output),
                },
            )
            higgs._validate_success(worker, response, output)
            print(
                f"source={source} reference_wav={wav} reference_sha256={reference_sha} "
                f"output={output} request_elapsed_sec={time.perf_counter() - started:.3f}",
                flush=True,
            )
    finally:
        higgs._shutdown_worker()
        print(
            f"total_elapsed_sec={time.perf_counter() - total_started:.3f}", flush=True
        )


if __name__ == "__main__":
    main()
