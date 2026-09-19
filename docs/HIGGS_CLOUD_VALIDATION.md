# Higgs Google Cloud validation gate

Production integration is implemented and offline-validated. Real Higgs GPU
runtime validation is still required. Higgs is the configured primary backend,
but the checked-in paths are placeholders and it must not be treated as deployable
until every required checkbox below passes and its evidence is reviewed.

## Evidence boundary

Already proven by CPU/offline tests:

- canonical validation, Higgs control resolution, worker lifecycle, request
  construction, FFmpeg rate processing, assembly, telephone output, metadata,
  QC, batch/resume identity, and explicit speaker-reference materialization;
- no automatic Higgs-to-CosyVoice fallback;
- `slow`/`fast` use FFmpeg `atempo=0.85`/`1.15`; SGLang speed remains `1.0`;
- `pause_before` is assembly-owned: `none`/`short`/`long` produce
  `0`/`500`/`900 ms`; `pause_after` is fixed at `0`, and turn WAVs contain
  speech only.

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

## Evidence baseline

Collect final Cloud validation evidence only after this work is merged to
`main`. Check out the exact merged `main` commit and record its full SHA in the
evidence package. Evidence collected from an unmerged feature branch is not the
final production baseline.

## Phase A — environment identity

- [ ] Record the exact merged `main` SHA, config SHA, and backend identity.
- [ ] Record model ID, local checkpoint path, revision if known, and available
      artifact hashes; do not commit model files.
- [ ] Record Python, SGLang-Omni, SGLang, Torch, Torch CUDA-build marker, CUDA
      runtime/driver, and GPU model/memory.
- [ ] Confirm the GPU is visible and the checkpoint is present.
- [ ] Confirm `ffmpeg` and the configured `sgl-omni` executable are available.
- [ ] Confirm reference assets exist and output paths are writable.

## Higgs reference approval — mandatory before production materialization

For every production `speaker_id`, use this workflow:

```text
candidate reference WAV
  -> record exact path + SHA-256
  -> review-only registry copy (candidate) or production registry (approved)
  -> materialize speaker assignments with --backend higgs
  -> generated speaker sidecar
  -> 5703tts --speaker-sidecar ...
  -> production Higgs synthesis
  -> perceptual and technical review
```

The registry and materializer are preprocessing inputs. The renderer does not
look up the registry at runtime: it reads the production speaker ID and exact
Higgs reference path/SHA only from the generated sidecar. The `tts.higgs`
configuration contains runtime/model settings only and never supplies speaker
references.

Checklist:

- [ ] Choose a candidate reference WAV and record its exact path and SHA-256.
- [ ] For an unapproved candidate, put the exact object in a review-only copy
      of the registry used solely to materialize validation input; do not
      change the production registry or treat the candidate as approved.
- [ ] Materialize the validation sidecar with `--backend higgs` and confirm it
      contains the expected `spk_*`, reference path, and SHA-256.
- [ ] Run `5703tts` with that generated `--speaker-sidecar` through the
      production worker path.
- [ ] Verify the intended voice identity and clear separation from other
      production speakers.
- [ ] Review audible quality, clipping, noise, truncation, and other artifacts.
- [ ] Record reviewer, date, result, and evidence paths.
- [ ] Only after acceptance, add this exact object to the production registry
      and regenerate any production sidecars that will use it:

```json
"higgs_reference": {
  "reference_wav": "path/to/approved.wav",
  "sha256": "<exact 64-character SHA-256>"
}
```

An existing `primary_reference.prompt_wav` may be evaluated as a candidate, but
candidate does not mean approved. Never populate or approve `higgs_reference`
merely because the same physical WAV already works with CosyVoice; the two
reference types are not interchangeable.

## Phase B — production worker/server startup

- [ ] Launch through `5703tts`/`backends/higgs_worker.py`, not a manual SGLang-only call.
- [ ] Confirm `/health` readiness, record cold-start time, and confirm no OOM.
- [ ] Confirm worker/SGLang stderr drains continuously without deadlock.
- [ ] Stop normally and confirm the worker and server process group exit and GPU
      allocation is released.

## Phase C — one normal turn

- [ ] Use an explicitly approved reference and a valid production turn with all
      required controls, for example `rate=normal`, `arousal=2`,
      `affect=neutral`, `pause_before=none`, `pause_within=0`, and
      `hesitations=0`.
- [ ] Confirm the reference reaches SGLang and the request returns HTTP 200 WAV.
- [ ] Validate the final turn WAV and record sample rate, channels, frames,
      duration, file size, and hash.
- [ ] Confirm metadata carries the expected backend/mapping/reference identity.
- [ ] Confirm no FFmpeg rate operation is required or reported as executed.

## Phase D — controlled-turn matrix

Every matrix row must be a valid production turn containing all six required
controls. Render at least:

| Case | `rate` | `arousal` | `affect` | `pause_before` | `pause_within` | `hesitations` | Expected focus |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | `normal` | `2` | `neutral` | `none` | `0` | `0` | no rate postprocess; 0 ms leading pause |
| Low + warm | `slow` | `1` | `warm` | `short` | `0` | `1` | contentment override; `atempo=0.85`; 500 ms leading pause |
| High + angry | `fast` | `3` | `angry` | `long` | `1` | `2` | `atempo=1.15`; 900 ms leading pause; native pause planning |
| Sad | `normal` | `1` | `sad` | `none` | `0` | `1` | sad mapping and one lexical hesitation |
| Anxious | `normal` | `3` | `anxious` | `short` | `1` | `2` | anxious mapping, native pause planning, and two hesitations |

Together these rows cover `slow`/`normal`/`fast`, arousal `1`/`2`/`3`, all
five production affects, every `pause_before` value, `pause_within > 0`, and
hesitation counts `1` and `2` without constructing an invalid partial turn.

For each case, record the exact prefix/model input, returned WAV metadata,
postprocess result, timings, and metadata JSON. A successful render proves
technical execution only; it does not prove perceptual monotonicity or mapping
quality.

## Phase E — two-speaker dialogue

- [ ] Materialize with `--backend higgs` using two explicitly approved registry
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

Higgs must not pass the real-runtime production gate while any of these remain unresolved:

- unreliable model/server startup, OOM, or stderr deadlock;
- failed reference-conditioned requests or wrong speaker/reference selection;
- unacceptable speaker separation, quality, clipping, noise, or artifacts;
- orphaned server process/GPU allocation or batch resource growth;
- failed rate postprocessing or metadata that disagrees with applied controls;
- resume across backend/reference identity, systematic QC failure, or
  unacceptable project-level perceptual quality.

## Evidence package

Keep a small reviewable package containing the exact merged `main` commit SHA,
config SHA/backend identity, environment/package/GPU identity,
checkpoint/reference identifiers, startup and per-turn timings, WAV metadata,
shutdown result, QC output, representative metadata JSON, batch summary, and
failures/warnings. Do not commit checkpoints, generated corpora, caches,
secrets, or routine full server logs.

Backend selection remains explicit. There is no automatic fallback. Any future
fallback must record `requested_backend`, `actual_backend`, `fallback_used`, and
`fallback_reason`; that policy is not implemented here.
