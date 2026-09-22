# Production architecture

This is the design summary. See [DATA_CONTRACTS.md](DATA_CONTRACTS.md) for persisted/internal shapes, [CODE_READING_GUIDE.md](CODE_READING_GUIDE.md) for source order, and [CURRENT_STATUS.md](CURRENT_STATUS.md) for claims about evidence and readiness.

## Purpose and backend policy

`5703tts` accepts only final nested JSON/JSONL from `upstream/Controlled-TTS-v1` and produces reviewable dialogue audio plus provenance, QC, and batch state.

- Higgs is the production-primary backend.
- CosyVoice3 is an explicitly selected backup/secondary backend.
- `tts.engine` selects one backend for the complete run.
- There is no automatic fallback.
- Final source records are never rewritten with production speakers or plans.

## End-to-end model

```text
final JSON / JSONL
  -> InputRecord (canonical source SHA)
  -> strict final contract validation
  -> known-issue exclusion
  -> speaker sidecar
  -> CanonicalDialogue / CanonicalTurn
  -> backend-specific preparation
  -> PreparedDialogue / PreparedTurn
  -> whole-dialogue preflight
  -> sequential backend synthesis
  -> TurnRenderResult + speech-only turn WAVs
  -> assembly + speech timestamps
  -> clean WAV
  -> telephone WAV
  -> provenance metadata
  -> structural/control-integrity QC
  -> batch manifest v2
  -> semantic resume + artifact-integrity validation
```

## Source, canonical, prepared, and result ownership

`InputRecord` owns a defensive source snapshot. `record_sha256` hashes canonical JSON content rather than path, line number, or formatting. Strict parsing rejects duplicate keys, NaN/Infinity, non-object records, and blank IDs; a malformed JSONL row becomes a record-level failure so later rows survive.

`CanonicalTurn` owns backend-neutral source/render semantics: text, labels, upstream role, logical role, scenario speaker ID, production render speaker ID, raw requests, and normalized controls. It owns no model-native execution field.

Prepared turns own the complete exact backend execution plan. Constructors validate and deep-copy plans; plan access is defensive. Execution, metadata, and QC consume that cached plan and do not remap controls.

`TurnRenderResult` owns only the link from prepared turn identity to its speech-only WAV and execution statuses. Assembly owns dialogue timing.

## Speaker and reference boundary

Upstream `User`/`Listener`, logical `caller`/`counsellor`, upstream scenario speaker IDs, and production `spk_*` render IDs are distinct.

The external sidecar binds source identities to render identities and selected references. Higgs requires `higgs_reference.reference_wav`; CosyVoice requires `cosyvoice_reference.prompt_wav` and `prompt_text`. Declared paths remain provenance; resolved paths serve runtime access; live SHA verification protects current bytes. A CosyVoice primary reference is not automatically an approved Higgs reference.

## Backend ownership

### Higgs

The main process communicates with a standard-library worker over JSON Lines. The worker owns and reaps one SGLang-Omni process group, routes server diagnostics through stderr, polls `GET /health`, and sends `POST /v1/audio/speech`. The parent continuously drains stderr and reuses the loaded server across turns. Reference conditioning is sent as `references: [{"audio_path": ...}]`.

`FROZEN_GENERATION_FIELDS` includes model speed `1.0`. Semantic rate is postprocessing: slow FFmpeg `atempo=0.85`, normal no transform, fast `atempo=1.15`. This code path records request/output structure and does not embed an external runtime-environment attestation in per-dialogue metadata. Project-level GPU runtime evidence is tracked in [CURRENT_STATUS.md](CURRENT_STATUS.md) and dedicated evidence artefacts. Runtime execution does not by itself establish production reference approval, reference-conditioning quality, control fidelity, or perceptual quality.

### CosyVoice3

The parent launches a worker with the CosyVoice-specific Python environment. The model loads once and is reused. Both zero-shot and instruct2 calls freeze `text_frontend=False`. Rate maps to `0.8/1.0/1.2`; affect/arousal instruction mapping is provisional; hesitation is lexical; pause-before is assembly-owned; positive pause-within is unsupported and fails closed.

## Preflight and orchestration boundary

`run_dialogue()` validates, prepares, then preflights every turn before creating the output directory or issuing the first synthesis request. This prevents a later invalid turn/reference from being discovered after earlier GPU work. Backend synthesis functions retain local defensive checks, but do not replace the whole-dialogue gate.

After preflight, turns render sequentially; assembly produces clean audio and timestamps; telephone audio is derived; metadata is written; structural QC runs; exceptions become batch-visible `PipelineResult` failures.

## Audio ownership

Turn WAVs contain speech only. Assembly inserts `pause_before` before timestamp start, validates exact ordinal/source-result alignment, and applies a small edge fade without crossfade. Clean audio is the assembled dialogue.

Telephone audio is only mono conversion, resampling, high-pass/low-pass filtering, and level reduction. It is not a codec, packet-loss/noise/room model, or complete PSTN simulation.

## Trust boundaries

1. Strict JSON parsing prevents ambiguous external objects and non-standard numbers.
2. Final-contract validation accepts one schema and never adapts legacy shapes.
3. Sidecar source SHA and speaker alignment prevent cross-record/materialization drift.
4. Selected references separate declared identity from resolved access and live bytes.
5. Prepared constructors defend source/role/control/backend-plan invariants.
6. Whole-dialogue preflight defends the first synthesis boundary.
7. Worker stdout is protocol-only; stderr is continuously consumed.
8. Dialogue output paths must be direct children and may not be per-dialogue symlinks.
9. Cleanup uses a managed-name allowlist and never follows metadata-provided paths.

## Semantic identity

The per-dialogue render fingerprint contains source dialogue ID and canonical record SHA, shared render-affecting configuration, selected backend static runtime/control identity, selected role/render-speaker/reference materialization, and exclusion decision.

It intentionally excludes input file location, JSON formatting, and unselected backend configuration/reference state. Static backend identity is used because fingerprinting happens before dialogue preparation. Metadata instead uses dialogue-aware backend identity with the exact references from the real prepared dialogue; fake dialogue objects are not used.

Configured runtime/model paths and control mapping are recorded, but the full checkpoint and runtime environment are not cryptographically hashed. This is a current reproducibility limitation.

## Resume boundary

Manifest v2 is authoritative batch state. Resume requires both a previous successful result with the same semantic fingerprint and live artifact/reference integrity.

Artifact checks cover metadata parsing, source/dialogue and backend/implementation identity, turn identity/order, labels/requested controls, timing sanity, expected filenames, readable non-empty turn/clean/telephone WAVs, and live selected-reference SHA. File existence alone is insufficient.

## Output ownership and rerender

A valid resume performs no cleanup. Every rerender first removes only pipeline-owned direct-child names, then calls `run_dialogue`. Cleanup failure prevents rendering into a mixed old/new set.

There is no recursive deletion, `rmtree`, or broad `*.wav` deletion. Unmanaged files and nested directories survive; a managed-name directory fails closed; a managed child symlink is unlinked without following it. Dialogue IDs cannot contain traversal separators or alias another output child. The output root itself may be symlinked.

## Manifest and exclusion

Manifest v2 records rendered, resumed, excluded, input-error, and render-failed outcomes. It is atomically written at batch completion, not checkpointed after every dialogue.

Known-issue exclusion is external policy data and occurs before sidecar/preparation requirements. Excluded records retain reason/provenance/policy SHA in the manifest but have no render output or fingerprint.

## Scope boundary

This document describes ownership and data flow, not runtime readiness or
perceptual evidence. Current evidence levels and remaining gates belong in
[CURRENT_STATUS.md](CURRENT_STATUS.md). Structural QC is intentionally not
perceptual QA, and the telephone-labelled output is intentionally not a full
telephone-network simulation.
