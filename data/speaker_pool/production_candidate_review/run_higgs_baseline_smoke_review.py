"""Review-only, same-text Higgs smoke for the ten active v0.1 speakers."""

import hashlib
import json
import time
from pathlib import Path

import yaml

from tts5703.backends import higgs

ROOT = Path(__file__).resolve().parents[3]
PACK = Path(__file__).resolve().parent
POOL = ROOT / "data/speaker_pool/vctk_v0.1"
EXPECTED_ACTIVE_IDS = (
    "spk_001",
    "spk_003",
    "spk_004",
    "spk_006",
    "spk_007",
    "spk_008",
    "spk_009",
    "spk_010",
    "spk_011",
    "spk_012",
)
TEXT = "I understand. Take your time and tell me what happened when you are ready."


def main() -> None:
    active = json.loads((POOL / "active_speakers.json").read_text(encoding="utf-8"))
    registry = json.loads((POOL / "speaker_registry.json").read_text(encoding="utf-8"))
    active_ids = active["active_speaker_ids"]
    excluded_ids = {
        entry["speaker_id"] for entry in active["excluded_from_active_pool"]
    }
    speakers = {entry["speaker_id"]: entry for entry in registry["speakers"]}
    if (
        tuple(active_ids) != EXPECTED_ACTIVE_IDS
        or len(speakers) != len(registry["speakers"])
        or set(active_ids) & excluded_ids
        or set(speakers) != set(active_ids) | excluded_ids
    ):
        raise ValueError(
            "Active v0.1 speakers do not match the registry and exclusions"
        )

    references = []
    for speaker_id in active_ids:
        reference = speakers[speaker_id]["primary_reference"]
        wav = ROOT / reference["prompt_wav"]
        if not wav.is_file():
            raise FileNotFoundError(f"Primary reference WAV missing: {wav}")
        with wav.open("rb") as handle:
            actual_sha = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual_sha != reference["sha256"]:
            raise ValueError(f"Primary reference WAV SHA-256 mismatch: {wav}")
        references.append((speaker_id, wav, actual_sha))

    config = yaml.safe_load(
        (ROOT / "config/config_higgs_cloud.yaml").read_text(encoding="utf-8")
    )["tts"]["higgs"]
    output_dir = PACK / "higgs_baseline_smoke_review"
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
        for speaker_id, wav, reference_sha in references:
            output = output_dir / f"{speaker_id}.wav"
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
                f"speaker_id={speaker_id} reference_wav={wav} "
                f"reference_sha256={reference_sha} output={output} "
                f"request_elapsed_sec={time.perf_counter() - started:.3f}",
                flush=True,
            )
    finally:
        higgs._shutdown_worker()
        print(
            f"total_elapsed_sec={time.perf_counter() - total_started:.3f}", flush=True
        )


if __name__ == "__main__":
    main()
