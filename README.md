# 5703tts

`5703tts` is the final dialogue-rendering stage of the COMP5703 / CS-28
Controlled-TTS workflow. It turns final nested JSON/JSONL dialogues into
speech-only turn WAVs, assembled clean and telephone-labelled WAVs, provenance
metadata, structural QC, and manifest-v2 batch state.

```text
InputRecord -> validation/exclusion -> speaker sidecar -> canonical/prepared plan
-> whole-dialogue preflight -> one backend -> assembly -> metadata/QC -> manifest/resume
```

## Current boundary

At baseline `8347253aff79aa542fcf19278cc1b80f0d2a816c` (2026-09-22), both
Higgs3 and CosyVoice3 have achieved real Google Cloud GPU runtime execution.
That does not establish production reference approval, control/perceptual
fidelity, production-scale rendering, or clinical usefulness.

Higgs3 is the configured default backend. CosyVoice3 is explicitly selected as
the secondary backend. Set `tts.engine` to choose one backend per run; there is
no automatic fallback.

## Input and output

Input is one final nested dialogue object in `.json`, or one object per nonblank
line in `.jsonl`. Required controls are `rate`, `arousal`, `affect`,
`pause_before`, `pause_within`, and `hesitations`. Source `User`/`Listener`
roles and scenario IDs remain unchanged; a sidecar maps logical roles to
production `spk_*` identities and backend-specific references.

## Fresh Cloud quick start

```bash
bash scripts/setup_cloud_environment.sh
bash scripts/setup_cloud_environment.sh --check
```

After bootstrap, a Higgs render uses the generated
`config/config_higgs_cloud.yaml`:

```bash
uv run 5703tts \
  --input data/final/dialogues.jsonl \
  --speaker-sidecar data/speaker_sidecar.json \
  --config config/config_higgs_cloud.yaml \
  --output data/output \
  --resume
```

`config/config.yaml` is a repository template with placeholder Higgs paths.
For an explicit CosyVoice run, use `config/config_cosyvoice.yaml`. See
[RUNNING.md](docs/RUNNING.md) for sidecar materialization, smoke tests, batch,
resume, and troubleshooting.

## Documentation

- [Pipeline](docs/PIPELINE.md) — architecture, contracts, controls, outputs, source map, and invariants.
- [Running](docs/RUNNING.md) — offline checks, Cloud bootstrap, backend runs, batch, resume, and troubleshooting.
- [Status and validation](docs/VALIDATION.md) — evidence levels, current runtime status, validation checklist, and remaining gates.
- [VCTK speaker pool](data/speaker_pool/vctk_v0.1/README.md) — local speaker-pool and reference asset contract.

Development checks:

```bash
uv run ruff check .
uv run --with pytest pytest -q
git diff --check
```
