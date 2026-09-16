# Higgs Google Cloud validation gate

Production integration is implemented and offline-validated. Real Higgs GPU
runtime validation is still required. Do not make Higgs the default backend
until every required checkbox below passes and its evidence is reviewed.

## Evidence boundary

Already proven by CPU/offline tests:

- canonical validation, Higgs control resolution, worker lifecycle, request
  construction, FFmpeg rate processing, assembly, telephone output, metadata,
  QC, batch/resume identity, and explicit speaker-reference materialization;
- no automatic Higgs-to-CosyVoice fallback;
- `slow`/`fast` use FFmpeg `atempo=0.85`/`1.15`; SGLang speed remains `1.0`;
- `pause_before_ms` and `pause_after_ms` remain assembly-owned.

Still requiring real GPU evidence:

- production-worker startup against the chosen cloud stack;
- reference-conditioned synthesis and speaker identity;
- perceptual quality and controlled-turn behaviour;
- long-running cleanup, resource stability, and batch performance.

## Frozen experiment identity

The frozen experiment at commit
`7b005135e148fad8a142f39dc8be3144c3e5adcc` successfully completed one
unconditioned technical smoke with:

| Item | Experiment evidence |
| --- | --- |
| Model | `bosonai/higgs-tts-3-4b` |
| Checkpoint revision | `0056125158f940389ab0808a581b8b2c590b32d4` |
| `model.safetensors` SHA-256 | `2f7965264c360b38180885006944aa16bd1de20f4e6cff79f6473bfcf8ae3d5a` |
| SGLang-Omni / SGLang | `0.1.3` / `0.5.16` |
| Python / Torch | `3.12.11` / `2.11.0` |
| Torch CUDA build marker | `13.0` |
| GPU | NVIDIA L4 |
| HTTP endpoints | `GET /health`, `POST /v1/audio/speech` |
| Observed output | 24 kHz mono WAV |

These are experiment observations, not permanent production version
requirements. A newer pinned stack is acceptable, but its exact versions,
checkpoint identity, GPU, request result, and output metadata become a new
evidence set. The old smoke did not verify reference-conditioned synthesis.

## Phase A — environment identity

- [ ] Record the exact `5703tts` Git commit and config SHA/backend identity.
- [ ] Record model ID, local checkpoint path, revision if known, and available
      artifact hashes; do not commit model files.
- [ ] Record Python, SGLang-Omni, SGLang, Torch, Torch CUDA-build marker, CUDA
      runtime/driver, and GPU model/memory.
- [ ] Confirm the GPU is visible and the checkpoint is present.
- [ ] Confirm `ffmpeg` and the configured `sgl-omni` executable are available.
- [ ] Confirm reference assets exist and output paths are writable.

## Higgs reference approval — mandatory before production materialization

For every production `speaker_id`:

- [ ] Choose a candidate reference WAV and record its exact path and SHA-256.
- [ ] Use a temporary explicit Higgs config to run real reference-conditioned
      synthesis through the production worker path.
- [ ] Verify the intended voice identity and clear separation from other
      production speakers.
- [ ] Review audible quality, clipping, noise, truncation, and other artifacts.
- [ ] Record reviewer, date, result, and evidence paths.
- [ ] Only after acceptance, add this exact object to that speaker's registry
      entry:

```json
"higgs_reference": {
  "reference_wav": "path/to/approved.wav",
  "sha256": "<exact 64-character SHA-256>"
}
```

The existing `primary_reference.prompt_wav` may be tested as a candidate:

```text
CosyVoice primary prompt
  -> temporary explicit Higgs test config
  -> real Higgs synthesis and review
  -> accepted: pin the same path + SHA under higgs_reference
  -> rejected: select or create a separate Higgs reference
```

Candidate does not mean approved. Never populate `higgs_reference` merely
because the same physical WAV already works with CosyVoice.

## Phase B — production worker/server startup

- [ ] Launch through `5703tts`/`higgs_worker.py`, not a manual SGLang-only call.
- [ ] Confirm `/health` readiness, record cold-start time, and confirm no OOM.
- [ ] Confirm worker/SGLang stderr drains continuously without deadlock.
- [ ] Stop normally and confirm the worker and server process group exit and GPU
      allocation is released.

## Phase C — one normal turn

- [ ] Use an explicitly approved reference and `rate=normal` with absent affect
      controls or one simple supported case.
- [ ] Confirm the reference reaches SGLang and the request returns HTTP 200 WAV.
- [ ] Validate the final turn WAV and record sample rate, channels, frames,
      duration, file size, and hash.
- [ ] Confirm metadata carries the expected backend/mapping/reference identity.
- [ ] Confirm no FFmpeg rate operation is required or reported as executed.

## Phase D — controlled-turn matrix

Render at least:

- [ ] slow + neutral + medium (`atempo=0.85`);
- [ ] fast + neutral + medium (`atempo=1.15`);
- [ ] low + warm (contentment override);
- [ ] sad, anxious, angry, and warm;
- [ ] absent arousal and affect.

For each case, record the exact prefix/model input, returned WAV metadata,
postprocess result, timings, and metadata JSON. A successful render proves
technical execution only; it does not prove perceptual monotonicity or mapping
quality.

## Phase E — two-speaker dialogue

- [ ] Materialize with `--higgs-ready` using two explicitly approved registry
      entries and matching persistent speaker IDs.
- [ ] Confirm each turn uses the correct reference and one server is reused.
- [ ] Confirm all turn WAVs, clean assembly, timestamps, telephone output,
      metadata, and QC.
- [ ] Listen for speaker confusion and confirm audible speaker separation.

## Phase F — small batch and resume

- [ ] Run multiple dialogues through one persistent worker/model.
- [ ] Record startup and per-turn inference times plus the batch summary.
- [ ] Check process/file-descriptor counts and GPU memory for obvious growth.
- [ ] Confirm an identical completed run resumes without worker startup.
- [ ] Confirm changed config/backend/reference bytes rerender.
- [ ] Confirm CosyVoice output never resumes as Higgs, or vice versa.
- [ ] Confirm final worker/server shutdown releases processes and GPU allocation.

## Failure gates

Higgs must not become the default backend while any of these remain unresolved:

- unreliable model/server startup, OOM, or stderr deadlock;
- failed reference-conditioned requests or wrong speaker/reference selection;
- unacceptable speaker separation, quality, clipping, noise, or artifacts;
- orphaned server process/GPU allocation or batch resource growth;
- failed rate postprocessing or metadata that disagrees with applied controls;
- resume across backend/reference identity, systematic QC failure, or
  unacceptable project-level perceptual quality.

## Evidence package

Keep a small reviewable package containing the Git commit, config SHA/backend
identity, environment/package/GPU identity, checkpoint/reference identifiers,
startup and per-turn timings, WAV metadata, shutdown result, QC output,
representative metadata JSON, batch summary, and failures/warnings. Do not
commit checkpoints, generated corpora, caches, secrets, or routine full server
logs.

Backend selection remains explicit. There is no automatic fallback. Any future
fallback must record `requested_backend`, `actual_backend`, `fallback_used`, and
`fallback_reason`; that policy is not implemented here.
