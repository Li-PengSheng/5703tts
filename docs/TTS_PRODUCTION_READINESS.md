# TTS production readiness

## Current production contract

The production CLI accepts only final nested JSON/JSONL from
`upstream/Controlled-TTS-v1`. Higgs is primary and CosyVoice3 is an explicitly
selected backup. Both consume the same `InputRecord -> CanonicalDialogue` path and
share assembly, metadata, QC, manifest v2, and resume. There is no automatic fallback.

Legacy flat input, schema v0.2, Kokoro, and the normalized legacy renderer remain only
as unreachable U3 cleanup inventory.

## Offline-verified behavior

- Strict JSON/JSONL parsing and record-level malformed-line reporting.
- Final contract validation and immutable source records.
- Backend-neutral canonical turns and production `spk_*` assignment.
- Frozen Higgs mapping parity and one synthesis call per turn.
- Cached CosyVoice speed/instruction/prompt requests without `NormalizedTurn`.
- Selected-backend-only reference requirements and SHA verification.
- Shared pause-before, speech-only turn files, ordering, and timing.
- Backend-specific planned metadata and structural/control-integrity QC.
- Manifest v2, semantic fingerprints, live-reference resume checks, and atomic writes.
- Explicit exclusion before speaker/preparation requirements.
- No automatic backend fallback.

## Speaker materialization

Use an explicit target:

```bash
uv run python scripts/materialize_speaker_assignments.py \
  --input data/final/dialogues.jsonl \
  --assignments data/speaker_assignments.jsonl \
  --registry data/speaker_pool/vctk_v0.1/speaker_registry.json \
  --active-speakers data/speaker_pool/vctk_v0.1/active_speakers.json \
  --output data/materialized-unused \
  --config-out config/materialized.yaml \
  --base-config config/config.yaml \
  --manifest data/speaker_sidecar.json \
  --backend both
```

`higgs` requires only approved Higgs references, `cosyvoice` requires only primary
CosyVoice references, and `both` requires both. Final source records are never
rewritten and reference types are never substituted.

## Known capability boundary

CosyVoice `pause_within > 0` fails closed before output creation because no verified
deterministic realization is available. Instruction-based arousal/affect mapping is
provisional. Offline tests do not prove perceptual fidelity.

## Next runtime gate

Real cloud/GPU validation must establish Higgs startup, reference conditioning,
speaker separation, intelligibility, stability, performance, and operational cleanup.
Until that evidence exists, the repository claims software-path readiness only—not
acoustic quality or production capacity.
