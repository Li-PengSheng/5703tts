# Historical TTS production-readiness snapshot

> Historical evidence snapshot. For current status see [../CURRENT_STATUS.md](../CURRENT_STATUS.md).

All status statements below refer to that historical snapshot, not the current
project state.

## Current decision

This document preserves an earlier evidence snapshot. It predates the current
project-level report that real Higgs3 and CosyVoice3 GPU runtime execution has
been achieved. It remains useful for the evidence claims recorded at that time;
it is not the current-state source of truth.

Higgs remains the configured production primary. CosyVoice3 remains an explicitly selected backup/secondary. There is no automatic fallback.

## Software/offline evidence

The test suite covers:

- strict JSON/JSONL parsing, canonical record hashes, and isolated malformed rows;
- final-contract-only validation and immutable-ish source snapshots;
- source/logical/scenario/render speaker identity separation;
- selected-backend-only reference contracts and live SHA verification;
- canonical/prepared boundaries, frozen plan validation, and defensive copies;
- frozen Higgs mapping parity, one synthesis call per turn, request construction, worker lifecycle, and FFmpeg rate processing;
- cached CosyVoice prompt/speed/instruction requests and `text_frontend=False`;
- whole-dialogue preflight before output directory/GPU work;
- speech-only turn files, pause-before assembly, ordering, timestamps, clean/telephone outputs;
- requested/planned/executed metadata and structural/control-integrity QC;
- manifest v2, semantic fingerprinting, artifact/reference integrity resume, path security, and managed stale cleanup;
- explicit known-issue exclusion before sidecar/preparation requirements;
- absence of automatic backend fallback.

This evidence validates software behavior, not acoustic or perceptual quality.

## CosyVoice3 runtime evidence

Phase 3A real runtime evidence exists for the backup backend:

- 5 dialogues;
- 30 turns;
- 0 failures;
- identical resume 5/5;
- controlled artifact corruption triggered rerender;
- stale turn cleanup passed while unmanaged content survived.

The raw runtime log remains in the repository, but the evidence summary above—not the log volume—is the handoff statement. This evidence applies to CosyVoice3 and does not validate Higgs. CosyVoice affect/arousal instruction mapping remains provisional and `pause_within > 0` remains unsupported/fail-closed.

## Higgs status

Implemented and offline-validated:

- selected approved-reference path/SHA contract;
- frozen Controlled-TTS-v1 plan and generation fields;
- parent/JSON-lines worker/SGLang ownership model;
- `/health` and `/v1/audio/speech` request handling;
- reference payload construction;
- worker reuse, stderr draining, fatal/recoverable error handling, process-group cleanup;
- atomic raw-WAV publication and slow/normal/fast rate lifecycle;
- common assembly, metadata, QC, manifest, resume, and cleanup integration.

Still pending on real Cloud hardware:

- actual chosen SGLang/Higgs stack startup and protocol compatibility;
- reference-conditioned synthesis with reviewed candidates;
- speaker identity/separation and perceptual review;
- intelligibility, naturalness, artifacts, and controlled-turn behavior;
- batch performance, long-running resource stability, and shutdown/GPU release.

Use [HIGGS_CLOUD_VALIDATION.md](../HIGGS_CLOUD_VALIDATION.md) as the formal evidence checklist and [HIGGS_PRODUCTION_RUNBOOK_CN.md](../HIGGS_PRODUCTION_RUNBOOK_CN.md) as its execution guide.

## Evidence boundary

Structural QC verifies identity, plan/control consistency, timing, execution status, and WAV readability. It does not establish emotion accuracy, acoustic arousal calibration, speaker similarity, naturalness, intelligibility, or clinical validity.

Telephone output is mono/resampling/high-pass/low-pass/level reduction only. It is not a telephony codec, packet-loss or line-noise model, room simulation, or complete PSTN simulation.

## Reproducibility boundary

Metadata/manifest record source, assignment, registry, active-speaker, mapping, implementation, config, selected reference, and configured backend runtime identity. The complete model checkpoint and runtime environment are not yet cryptographically pinned by backend identity; Cloud evidence must separately record model/runtime/package/CUDA/GPU identity.

Manifest v2 is atomically written at batch completion, not checkpointed after every dialogue. Interrupted work therefore relies on the next run's normal semantic and artifact-integrity decisions rather than a per-dialogue journal.

## Remaining production gates

1. Review and approve actual Higgs reference candidates.
2. Complete the full Higgs Cloud checklist against the exact merged `main` SHA.
3. Review technical and perceptual evidence before approving Higgs production use.
4. Receive the real final source corpus; it is not committed in this repository.
5. Run a large-scale production rehearsal on that final corpus and review failures, throughput, storage, resume, and handoff artifacts.

At the time of this historical snapshot, the accurate claim was: software/offline path strongly validated; CosyVoice mini-batch runtime demonstrated; Higgs production integration implemented but real reference-conditioned Cloud validation pending.

## Known limitations

- At snapshot time, real Higgs Cloud validation was pending.
- Higgs reference candidates need actual approval.
- Full model/runtime identity is not cryptographically pinned.
- Batch result is written at batch completion rather than after each dialogue.
- Structural QC is not perceptual QA.
- Telephone processing is not a real phone codec or PSTN simulation.
- CosyVoice positive pause-within is unsupported.
- CosyVoice affect/arousal instruction mapping is provisional.
- There is no automatic backend fallback.
- The final large source corpus is not committed.
