# Current status

This is the current-state source of truth for the repository. Snapshot date:
2026-09-22. Baseline commit:
`582464cbe17eadc852044f3cc923be3065a58135`.

## Pipeline

The implemented path is:

```text
InputRecord
-> final validation
-> exclusion policy
-> speaker sidecar
-> CanonicalDialogue
-> one selected PreparedDialogue
-> whole-dialogue preflight
-> backend synthesis
-> TurnRenderResult
-> clean WAV + telephone-labelled WAV
-> metadata + structural QC
-> manifest v2 + resume
```

Source records are immutable snapshots. Canonical objects own backend-neutral
semantics and production speaker IDs; prepared objects own the exact selected
backend plan; execution results own speech-only turn WAV links.

## Evidence levels

| Level | Repository meaning | Current status |
| --- | --- | --- |
| Implemented | Code path exists and is guarded by tests or structural checks | Final nested input, sidecar, both backend adapters, assembly, metadata, QC, manifest v2, resume, and cloud bootstrap |
| Offline tested | CPU/static tests and fixtures exercise the behavior | Broad test suite covers contracts, planners, workers, orchestration, identity, cleanup, and setup checks |
| Runtime verified | A real GPU environment has executed the selected backend path | Higgs3 and CosyVoice3 real Google Cloud GPU execution are reported achieved at project level |
| Control/perceptual validated | Control fidelity, reference behavior, and listening/perceptual evidence are reviewed | Not complete for either backend as a general claim; CosyVoice arousal/affect mapping remains provisional |
| Production-scale validated | Representative corpus, resource/recovery, and handoff rehearsal pass | Not complete |

Runtime execution is therefore not equivalent to production approval, acoustic
control fidelity, perceptual quality, or clinical usefulness.

## Backend status and policy

`tts.engine` is either `higgs` or `cosyvoice`; one run selects one backend.
Higgs3 is the configured default backend. CosyVoice3 is an explicitly selected
secondary backend. There is no automatic fallback.

Higgs3 has implemented reference-conditioned request construction, a reusable
worker/server boundary, the frozen Controlled-TTS-v1 planner, and GPU runtime
execution. Formal gates remain: approved production references, control and
perceptual review, representative batch/resource evidence, and production-scale
rehearsal.

CosyVoice3 has implemented and runtime-executed its isolated Python worker,
prompt contract, numeric speed mapping, provisional arousal/affect instructions,
shared lexical hesitation planning, and assembly timing. Positive
`pause_within` is unsupported and fails closed before synthesis. General control
and perceptual validation remains incomplete.

## Speakers and references

The identity layers are deliberately separate:

```text
upstream role: User / Listener
logical role: caller / counsellor
scenario speaker: C001 / L001
production render speaker: spk_001 / spk_008
```

Assignments and references live in an external sidecar; source JSON is not
rewritten. Higgs consumes `higgs_reference.reference_wav` and CosyVoice3
consumes `cosyvoice_reference.prompt_wav` plus `prompt_text`. The registry's
CosyVoice `primary_reference` is not automatically an approved Higgs reference.
The current VCTK registry has no formally approved `higgs_reference` entries.
VCTK demographic fields are descriptive metadata only, not population claims.

## Cloud environment

`scripts/setup_cloud_environment.sh` is the fresh Ubuntu 22.04/24.04 GPU setup
entry point. Secure Boot must be disabled; an enabled Secure Boot state makes
bootstrap stop, and `--check` reports it as non-PASS/`WRONG_VERSION`. The script
does not modify firmware or security settings. It supports `--all`, `--base`,
`--cosyvoice`, `--higgs`, and the read-only `--check` mode. It checks or installs the project Python 3.11
environment, CosyVoice Python 3.10 and pinned source/model, and Higgs/SGLang
Python 3.12, CUDA 13, UCX, model revision/SHA, local config, GPU access, and
port availability. Evidence runs must record the actual repository SHA and
runtime identities used; no historical development SHA is a fresh-install
requirement.

## Known limitations and remaining gates

- `runtime_verification: "not_runtime_verified"` remains in per-dialogue backend metadata. It means that metadata does not attest the external runtime environment; it does not mean the project has never run the backend.
- The full checkpoint/runtime environment is not cryptographically included in every render fingerprint.
- Structural QC checks identity, plan agreement, timing, execution, and WAV readability; it is not perceptual QA.
- The telephone-labelled WAV is mono conversion, resampling, high/low-pass filtering, and level reduction—not a complete PSTN simulation.
- Manifest v2 is written at batch completion, not as a per-dialogue journal.
- The final production-scale corpus and downstream clinical validation are outside this repository.
