# Status and validation

This is the single current status and evidence guide. Snapshot baseline:
`8347253aff79aa542fcf19278cc1b80f0d2a816c` (2026-09-22).

## Evidence levels

| Level | Meaning | Current project state |
| --- | --- | --- |
| Implemented | Code path exists and is guarded by tests or structural checks | Final input, sidecars, both backend adapters, assembly, metadata, QC, manifest v2, resume, and Cloud bootstrap |
| Offline tested | CPU/static tests and fixtures exercise the behavior | Broad repository suite covers contracts, planning, workers, orchestration, identity, cleanup, and setup checks |
| Runtime verified | A real GPU environment executed the selected backend path | Higgs3 and CosyVoice3 real Google Cloud GPU execution reported achieved at project level |
| Control/perceptual validated | Control fidelity, reference behavior, and listening evidence are reviewed | Not complete as a general claim for either backend; CosyVoice affect/arousal remains provisional |
| Production-scale validated | Representative corpus, resource/recovery, and handoff rehearsal pass | Not complete |

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
planner, and integrated assembly/metadata/QC paths. Formal gates remain:
production Higgs reference approval, systematic control/perceptual review,
representative batch/resource evidence, and production-scale rehearsal.

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
CosyVoice references are different contracts. The current VCTK registry has no
production-approved `higgs_reference` entries; runtime execution does not change
that status. VCTK demographics are descriptive metadata, not population claims.

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

Real Google Cloud GPU runtime execution has been achieved. The remaining
validation gates are:

- approve production Higgs references through the registry/sidecar workflow;
- review reference-conditioned speaker identity and separation;
- collect requested-versus-realized control and systematic perceptual evidence;
- run representative multi-dialogue batch, resource, recovery, and shutdown
  checks;
- rehearse production-scale rendering and handoff on the final corpus.

Do not describe these gates as “Higgs runtime pending”: runtime has occurred;
the pending work is formal validation beyond runtime execution.

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
- The final production-scale corpus and downstream clinical validation are
  outside this repository.
- There is no automatic backend fallback.

## Next acceptance gates

The current gates are limited to formal Higgs reference approval, systematic
control/perceptual validation, representative batch/resource/recovery evidence,
and production-scale rehearsal. Runtime execution itself is no longer a gate.
