# Status and validation

This is the current status and evidence guide at baseline
`628abd58c7c59850b7aeb5e5b12fd6047010053d` (2026-09-26).

## Evidence levels

| Level | Meaning | Current project state |
| --- | --- | --- |
| Implemented | Code path exists and is guarded by tests or structural checks | Final input, sidecars, both backend adapters, assembly, metadata, QC, manifest v2, resume, and Cloud bootstrap |
| Offline tested | CPU/static tests and fixtures exercise the behavior | Broad repository suite covers contracts, planning, workers, orchestration, identity, cleanup, and setup checks |
| Runtime verified | A real GPU environment executed the selected backend path | Higgs3 and CosyVoice3 real Google Cloud GPU execution reported achieved at project level |
| Control/perceptual validated | Control fidelity, reference behavior, and listening evidence are reviewed | Not complete as a general claim for either backend; CosyVoice affect/arousal remains provisional |
| Production-scale validated | Representative corpus, resource/recovery, and handoff rehearsal pass | Targeted 30-dialogue / 473-turn pilot and retry recovery validated; full 1000-dialogue production incomplete |

Runtime execution is not production reference approval, acoustic control
fidelity, perceptual quality, production-scale validation, or clinical
usefulness.

## Current status

### Pipeline and cloud

The implemented path is:

```text
InputRecord -> validation -> exclusion -> sidecar -> CanonicalDialogue
-> PreparedDialogue -> whole-dialogue preflight -> backend synthesis
-> TurnRenderResult -> clean/telephone WAV -> metadata/QC -> manifest/resume
```

The fresh Google Cloud entry point is
`scripts/setup_cloud_environment.sh`. It targets Ubuntu 22.04/24.04, requires
Secure Boot disabled, and checks the separate project 3.11, CosyVoice 3.10,
and Higgs/SGLang 3.12 environments plus CUDA/UCX, model revisions/SHA, GPU
availability, and local Higgs configuration.

### Backend policy

`tts.engine` selects `higgs` or `cosyvoice` for the entire run. Higgs3 is the
configured default backend; CosyVoice3 is an explicitly selected secondary
backend. There is no automatic fallback.

Higgs3 has real Google Cloud GPU runtime execution, reference-conditioned
request construction, worker/server ownership, the frozen Controlled-TTS-v1
planner, approved v0.2 references, frozen v0.2 assignments, and real-audio
validated termination QC. Frozen production runtime: `bosonai/higgs-tts-3-4b`,
SGLang-Omni 0.1.3 / SGLang 0.5.16 on NVIDIA L4;
`max_total_tokens=71680`, `max_running_requests=1`, `cuda_graph_max_bs=1`,
`max_new_tokens=1024`, `temperature=0.8`, `top_p=0.8`, `top_k=30`,
`stream=false`, `speed=1.0`. General control/perceptual review and full-corpus
production remain open.

CosyVoice3 has real Google Cloud GPU runtime execution through its isolated
worker, prompt contract, numeric speed mapping, provisional arousal/affect
instructions, shared hesitation planning, and assembly timing. Positive
`pause_within` remains unsupported and fails closed before synthesis. General
control and perceptual validation remains incomplete.

### Speakers and references

```text
User / Listener -> caller / counsellor -> C001 / L001 -> spk_*
```

Source JSON remains unchanged by assignment/materialization. Higgs and
CosyVoice references are different contracts. The frozen
`data/speaker_pool/vctk_v0.2/` pool has 15 approved Higgs references (commit
`4539983c8b8e7ecfe4eba4ff686c31083edeaba1`); the materialization gate
requires `production_approved` (commit
`acdb1acfdf6ada23ed036bd9306aaf1ed3f68824`). `spk_002` and `spk_005`
remain historical rejected/reserved IDs. The frozen
`data/speaker_assignment/v0.2/` assignment covers all 1000 dialogues with 9
caller and 6 counsellor render voices (commit
`43b5cc880e53e455e526dd82e05966d41dcbde9f`). Subsets select frozen rows.
VCTK source age does not acoustically realize corpus `age_band`.

Deterministic production chunk preparation is implemented and offline validated:
the builder selects canonical rows and matching frozen assignment rows into 10
chunks of 100, with SHA256 manifests and fail-closed publication. The real
frozen-data dry run produced 1000 dialogues and 15,934 turns. This is not GPU
validation and full-corpus rendering has not started.

## CosyVoice3 evidence

The retained production-path evidence covers:

- 5 dialogues and 30 turns;
- 0 failures;
- identical resume for 5/5 dialogues;
- corruption-triggered rerender;
- stale managed-turn cleanup while unmanaged content survived.

Additional control-integration evidence exercised 12 controlled cases with
readable, non-silent mono 24 kHz WAVs, including affect and arousal cases. Rate
direction was observed for one fixed text/reference (`slow=0.8`, `normal=1.0`,
`fast=1.2`). These are runtime/integration and directional mapping evidence,
not systematic listening or global acoustic validation.

## Higgs3 evidence

The canonical `data/final/corpus_v1_1000.jsonl` has 1000 dialogues, 15,934
turns, and SHA256
`137d74faf441a8e040fe8534b1e17439df541b51f2a3ae2b7e2bc64390027fb0`.
The latest representative pilot covered canonical IDs `corpus_v1_000000`
through `corpus_v1_000029`: 30 dialogues, 473 turns, all 15 production voices.
First pass accepted 28, quality-rejected 2, and had zero integrity failures
in 3780.519 s. `corpus_v1_000013` and `corpus_v1_000024` each hit a 40.68 s
`abnormal_tail`; both passed fresh controlled retry in 254.988 s. Effective
pilot acceptance is 30/30, with no observed OOM, worker crash, or reference
cache failure. The observed 2/30 first-pass rejection is only a small-pilot
observation, not a population reject-rate estimate.

Higgs quality sidecars are durable authority for assessed WAV bytes;
immutable/latest `batch_result` manifests are execution authority.
`quality_rejected.jsonl` is an operational queue (commit
`11d369835c729e024a3bd84d4103a77078091506`), and the controlled retry
builder is implemented (commit `e0770bf0e93f41332f6f7817ba7a5ac7c1a99ec1`).
`--resume` reuses or reassesses audio, never retries; a quality reject stays
terminal under same-fingerprint resume. Integrity failures are distinct, and
rejected evidence is preserved. Retry is operator-triggered fresh rendering
into a separate namespace, with no silent broad rerender.

`scripts/build_production_acceptance_manifest.py` combines first-pass and
controlled retry per-dialogue results as a derived delivery index. It does not
render, retry, or replace batch manifests or quality sidecars.
`--require-complete` is the all-corpus completion gate. For the real pilot,
the corresponding index counts are 28 `accepted_first_pass`, 2
`accepted_after_retry`, and 30 `accepted_total`. The full 1000-dialogue render
has not completed. General perceptual/control fidelity is not globally
validated. Historical `high + angry` Higgs control composition caused
speaker/gender identity instability in targeted tests; the current
`controlled_tts_v1` mapping has not been silently changed and still needs
explicit model/acoustic-control sign-off.

## Validation checklist

For a formal evidence run, record and review:

- [ ] exact merged repository SHA and config SHA;
- [ ] selected backend, mapping/implementation identity, and one-engine policy;
- [ ] model ID, revision/checkpoint SHA, Python/package/SGLang/UCX/CUDA/GPU identity;
- [ ] bootstrap `--check` result, executable/model paths, and local config;
- [ ] selected speaker references, declared/resolved paths, and live SHA-256;
- [ ] worker startup/health, per-turn output, cleanup, shutdown, and resource behavior;
- [ ] clean/telephone WAV metadata, structural QC, metadata, manifest, and resume result;
- [ ] reference identity, control behavior, artifacts, and reviewer outcome;
- [ ] representative batch and production-scale rehearsal results where claimed.

Only evidence actually collected for the current runtime identity may be marked
complete. A different model revision/checkpoint, SGLang or UCX version/commit,
CUDA identity, or other render-affecting runtime change creates a new runtime
identity and requires pins/evidence to be updated explicitly.

## Known limitations

- `runtime_verification: "not_runtime_verified"` in per-dialogue metadata means
  that metadata does not attest the external runtime environment; it does not
  mean the project has never executed the backend on a real GPU.
- Structural QC is not perceptual QA.
- Telephone output is not PSTN, codec, packet-loss, line-noise, or room simulation.
- CosyVoice affect/arousal mapping is provisional.
- Positive CosyVoice `pause_within` is unsupported and fails closed.
- The full checkpoint/runtime environment is not cryptographically included in
  every render fingerprint.
- Manifest v2 is written at batch completion, not as a per-dialogue journal.
- Full 1000-dialogue production and downstream clinical validation remain open.
- There is no automatic backend fallback.

## Next acceptance gates

The remaining gates are explicit `high + angry` model/acoustic sign-off,
systematic control/perceptual validation, and complete 1000-dialogue production
with a complete final acceptance manifest. Targeted pilot validation does not
establish those broader outcomes.
