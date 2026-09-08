# CosyVoice3 Production Control Mapping v1

## Decision

- Primary local rendering backend: CosyVoice3
- Model: `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
- Mapping: `cosyvoice3_control_mapping` version `v1`
- Implementation state: frozen for production integration
- Runtime fidelity status: `provisional`

The implementation is frozen, not universally acoustically proven. Rate has
directional real-audio evidence from the production path. Affect and arousal are
instruction-conditioned controls whose semantic and perceptual fidelity still needs
formal validation.

## Mapping

| Control | Semantic value | Backend realization | Evidence class | Known limitation |
| --- | --- | --- | --- | --- |
| Speaking rate | `slow` | Numeric `speed=0.8` | Exact adapter mapping; real model execution; directional production evidence | Not shown to be optimal for every text or reference speaker |
| Speaking rate | `normal` | Numeric `speed=1.0` | Exact adapter mapping; real model execution; directional production evidence | Not shown to be optimal for every text or reference speaker |
| Speaking rate | `fast` | Numeric `speed=1.2` | Exact adapter mapping; real model execution; directional production evidence | Not shown to be optimal for every text or reference speaker |
| Arousal | `low` | Low-activation instruction; no intentional rate request | Exact request mapping; instruction-conditioned; real synthesis succeeded | Perceptual arousal fidelity is not formally validated |
| Arousal | `medium` | Accepted with no additional arousal clause | Exact request mapping; real synthesis succeeded | Still selects `instruct2`; medium-only produces the generic-prefix edge case |
| Arousal | `high` | High-activation instruction; no intentional rate request | Exact request mapping; instruction-conditioned; real synthesis succeeded | Perceptual arousal fidelity is not formally validated |
| Coarse affect | `neutral` | Neutral, emotionally even conversational instruction | Exact request mapping; instruction-conditioned; real synthesis succeeded | Perceptual affect fidelity is not formally validated |
| Coarse affect | `sad` | Clearly sad and downcast instruction | Exact request mapping; instruction-conditioned; real synthesis succeeded | Perceptual affect fidelity is not formally validated |
| Coarse affect | `anxious` | Anxious, worried, uneasy instruction without panic or urgency | Exact request mapping; instruction-conditioned; real synthesis succeeded | Perceptual affect fidelity is not formally validated |
| Coarse affect | `angry` | Angry, firm, forceful instruction without shouting | Exact request mapping; instruction-conditioned; real synthesis succeeded | Perceptual affect fidelity is not formally validated |
| Coarse affect | `warm` | Warm, gentle, compassionate, supportive, reassuring instruction | Exact request mapping; instruction-conditioned; real synthesis succeeded | Perceptual affect fidelity is not formally validated |
| Coarse affect | `distressed` | Legacy distressed, worried, sad instruction | Exact request mapping; instruction-conditioned; compatibility smoke succeeded | Legacy compatibility only; not a sixth primary taxonomy label |
| Turn-boundary pause | Milliseconds before/after a turn | Deterministic pipeline timing during assembly | Pipeline control, not a model instruction | Does not define within-turn pause placement |

The exact clauses and composition rules live in
`src/tts5703/cosyvoice_controls.py`; this document does not replace that source of
truth. Rate-only and control-free turns use `zero_shot`. Any requested arousal or
coarse affect uses `instruct2`, including medium arousal when it adds no clause.

## Evidence

### Teammate/IG candidate and model evidence

The real mapping-calibration run
`week5_cv3_mapping_v0_t4_20260904_retry1` used an NVIDIA L4 and
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`. Its manifest records 22 targets, 22
successful targets, no failures, and 22 generated WAV files. Calibration v0 sent
slow as numeric speed `0.8` and fast as `1.2`. The run is mapping-calibration
evidence, not model-ranking or subjective-listening evidence.

Targeted Revision v1 subsequently defined the affect/arousal wording now used in
production and rate candidates `slow-A=0.85`, `slow-B=0.90`, `normal=1.00`, and
`fast=1.20`. Its handoff and progress record only CPU/static verification: no GPU,
model inference, or WAV generation was performed for that targeted phase. The
`0.85` and `0.90` values are therefore candidate design values, not validated
winners.

Reviewed read-only evidence:

- `testing/artifacts/week5_candidate_evaluation/week5_cv3_mapping_v0_t4_20260904_retry1/summary.md`
- `testing/artifacts/week5_candidate_evaluation/week5_cv3_mapping_v0_t4_20260904_retry1/results.jsonl`
- `testing/artifacts/week5_candidate_evaluation/week5_cv3_mapping_v0_t4_20260904_retry1/run_manifest.json`
- `testing/week5_candidate_evaluation/targeted_revision_v1/mappings/cosyvoice3.py`
- `testing/week5_candidate_evaluation/targeted_revision_v1/notes/A2_CV3_HANDOFF.md`
- `testing/week5_candidate_evaluation/targeted_revision_v1/notes/A5_SHARED_EXECUTION_PROGRESS_HANDOFF.md`

These paths are relative to the read-only extracted teammate repository and are not
production dependencies.

### LP production-integration evidence

Step 4A exercised the production route from validation through request resolution,
the CosyVoice worker, WAV generation, and control-resolution metadata on an NVIDIA
GeForce RTX 4060. All 12 controlled cases produced readable, non-silent, mono 24 kHz
WAV files. The five primary affect cases, all three arousal cases, and the legacy
`distressed` smoke completed successfully. This establishes integration and
synthesis success, not semantic fidelity.

For one fixed text and VCTK reference (`spk_001`, source speaker `p248`), the rate
durations were:

| Requested rate | Numeric speed | Duration |
| --- | ---: | ---: |
| `slow` | 0.8 | 13.24 s |
| `normal` | 1.0 | 10.00 s |
| `fast` | 1.2 | 9.30 s |

Both `fast < normal` and `normal < slow` were observed. This supports retaining the
current implementation but does not establish global optimality across speakers or
texts. The local evidence is retained under the ignored artifact run
`data/output/experiments/cv3_control_validation_v1_20260908T100933Z/`.

## Rate decision

Production remains `slow=0.8`, `normal=1.0`, and `fast=1.2`. These values have real
model execution evidence and the current production path showed the intended
duration direction. Targeted `0.85` and `0.90` slow values were never executed in
the targeted-revision phase, so they do not justify replacing `0.8`. The proposed
Step 4B comparison is cancelled and will not be duplicated here.

## Arousal

Low and high arousal use the targeted activation instructions and explicitly avoid
requesting a rate change. Medium is accepted and adds no arousal clause. Real
synthesis completed for all three, but acoustic and perceptual arousal fidelity
remains provisional.

## Coarse affect

The primary production labels are `neutral`, `sad`, `anxious`, `angry`, and `warm`.
Each resolves to its targeted instruction, and each synthesized successfully in
Step 4A. `distressed` remains accepted only for legacy compatibility; it is not a
sixth member of the proposed five-class taxonomy. Affect fidelity remains
provisional.

## Known limitations

- Affect semantic and perceptual fidelity is not formally validated.
- Arousal semantic and perceptual fidelity is not formally validated.
- Medium-only arousal selects `instruct2` with only the generic instruction prefix.
- A within-turn pause count does not specify exact placement.
- A hesitation count alone does not specify exact realization or placement.
- Control behaviour can vary with text and reference speaker.
- The mapping is implementation-frozen, not universally acoustically proven.

## Change policy

Do not change this production mapping without controlled validation evidence, an
agreed upstream-interface change, or a demonstrated production bug.
