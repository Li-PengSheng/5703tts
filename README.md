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

At baseline `628abd58c7c59850b7aeb5e5b12fd6047010053d` (2026-09-26),
Higgs production runtime, its 15 approved v0.2 references, frozen all-1000
speaker assignment, termination QC, and controlled retry workflow are validated.
A 30-dialogue / 473-turn pilot reached 30/30 acceptance after two fresh retries.
The full 1000-dialogue render and general control/perceptual sign-off remain open;
the historical `high + angry` identity issue requires explicit acoustic sign-off.
Deterministic 100-dialogue production chunks are prepared offline as described in
[RUNNING.md](docs/RUNNING.md).

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

After bootstrap, use the generated `config/config_higgs_cloud.yaml` and a
prepared small input with its matching speaker sidecar. See
[RUNNING.md](docs/RUNNING.md) for the render command. Render one prepared chunk
at a time; do not point `--input` at the full 1000-dialogue corpus.

`config/config.yaml` is a repository template with placeholder Higgs paths.
For an explicit CosyVoice run, use `config/config_cosyvoice.yaml`. See
[RUNNING.md](docs/RUNNING.md) for sidecar materialization, smoke tests, batch,
resume, and troubleshooting.

## Documentation

- [Pipeline](docs/PIPELINE.md) — architecture, contracts, controls, outputs, source map, and invariants.
- [Running](docs/RUNNING.md) — offline checks, Cloud bootstrap, backend runs, batch, resume, and troubleshooting.
- [Status and validation](docs/VALIDATION.md) — evidence levels, current runtime status, validation checklist, and remaining gates.
- [Current production handoff](docs/CloudProductionLog.md) — frozen runtime, pilot, retry, and open acoustic sign-off.
- [Higgs QC evidence](docs/evidence/HIGGS_AUDIO_QUALITY_VALIDATION.md) — targeted real-audio validation snapshot.
- [VCTK v0.2 production speaker pool](data/speaker_pool/vctk_v0.2/README.md) — approved references and asset contract.

Development checks:

```bash
uv run ruff check .
uv run --with pytest pytest -q
git diff --check
```
