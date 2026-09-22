# 5703tts

`5703tts` is the final dialogue-rendering stage of the COMP5703 / CS-28
Controlled-TTS workflow. It accepts final nested JSON/JSONL dialogues and
produces per-dialogue speech WAVs, a clean assembled WAV, a limited
telephone-labelled WAV, provenance metadata, structural QC, and a manifest-v2
batch state file.

```text
InputRecord -> final validation -> exclusion policy + speaker sidecar
-> CanonicalDialogue -> one selected backend PreparedDialogue
-> whole-dialogue preflight -> synthesis -> assembly -> metadata/QC
-> manifest v2 -> semantic resume
```

## Current boundary

The current baseline is commit `582464cbe17eadc852044f3cc923be3065a58135`
(snapshot: 2026-09-22). Project-level reports state that Higgs3 and CosyVoice3
have both achieved real Google Cloud GPU runtime execution. This does not by
itself establish reference approval, control/perceptual fidelity,
production-scale rendering, or downstream clinical usefulness.

Higgs3 is the configured default backend. CosyVoice3 is an explicitly selected
secondary backend. Set `tts.engine` to choose exactly one backend for a run;
there is no automatic Higgs-to-CosyVoice fallback.

## Input and output

Input is a `.json` file containing one dialogue or a `.jsonl` file containing one
dialogue object per nonblank line. The final contract requires `dialogue_id`,
`scenario.speakers`, turns, and the six required controls:
`rate`, `arousal`, `affect`, `pause_before`, `pause_within`, and `hesitations`.

The renderer keeps upstream `User`/`Listener` and scenario IDs unchanged. A
separate sidecar maps logical `caller`/`counsellor` roles to production
`spk_*` IDs and to the selected backend's reference contract.

## Start here

For a fresh Google Cloud / Ubuntu host, the checked-in bootstrap is the main
entry point:

```bash
bash scripts/setup_cloud_environment.sh
bash scripts/setup_cloud_environment.sh --check
```

Select a backend in config, provide a matching sidecar, then run:

```bash
uv run 5703tts \
  --input data/final/dialogues.jsonl \
  --speaker-sidecar data/speaker_sidecar.json \
  --config config/config_higgs_cloud.yaml \
  --output data/output \
  --resume
```

After the Cloud bootstrap, the Higgs command uses the generated
`config/config_higgs_cloud.yaml`. `config/config.yaml` is the repository
template/documented shape and its Higgs paths remain placeholders. For an
explicit CosyVoice run, use `config/config_cosyvoice.yaml`.

## Documentation map

- [Current status](docs/CURRENT_STATUS.md) — the only current-state source of truth.
- [Architecture](docs/ARCHITECTURE.md) — ownership boundaries and execution design.
- [Running](docs/RUNNING.md) — offline, Cloud, smoke, batch, resume, and troubleshooting.
- [中文安装与运行](docs/INSTALLATION_AND_SETUP_CN.md)
- [Data contracts](docs/DATA_CONTRACTS.md) — source, sidecar, prepared plans, metadata, manifest.
- [Controls and backends](docs/CONTROLS_AND_BACKENDS.md) — cross-backend realization and evidence level.
- [Code reading guide](docs/CODE_READING_GUIDE.md) — source order and call graph.
- [Development](docs/DEVELOPMENT.md) — environments, tests, invariants, and change checklist.
- [Higgs Cloud validation](docs/HIGGS_CLOUD_VALIDATION.md) and [Higgs runbook](docs/HIGGS_PRODUCTION_RUNBOOK_CN.md) — formal and operational evidence capture.
- [Historical evidence](docs/evidence/) and [archived guides](docs/archive/).

## Development checks

```bash
uv run ruff check .
uv run --with pytest pytest -q
git diff --check
```
