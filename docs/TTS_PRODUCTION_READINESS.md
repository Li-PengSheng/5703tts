# TTS Production Readiness

## Current status

The TTS renderer is production-ready under the current canonical v0.2
interface for controlled batch generation.

The final upstream adapter remains pending because the final nested
acoustic interface is not formally frozen.

This statement applies to the `5703tts` rendering workstream, not to the
whole CS-28 project.

## Current workflow

```
canonical v0.2 dialogues
    ->  scripts/assign_dialogue_speakers.py
speaker_assignments.jsonl
    ->  scripts/materialize_speaker_assignments.py
speaker-resolved render input + CosyVoice config
    ->  5703tts
turn WAVs + clean dialogue + telephone dialogue + metadata + QC
+ batch_result.json

later: 5703tts --resume
    ->  skip verified completed dialogues
        retry failed or incomplete dialogues
```

Assignment chooses persistent `speaker_id` values. Materialization resolves
those IDs into the renderer’s current `speaker` field and CosyVoice
`voice_map`. The renderer does not assign speakers.

Exact commands: [README](../README.md#speaker-assignment-and-materialization)
and the [demo](#reproducible-demo) below.

## Evidence table

| Stage | Evidence | Scale | Result | What it proves | What it does not prove |
| --- | --- | --- | --- | --- | --- |
| Control mapping integration | Offline tests of `cosyvoice_controls.py` / preflight | Mapping table + worker contract | Tests pass | Deterministic rate/arousal/affect mapping and instruct2 vs zero-shot routing | Perceptual fidelity |
| Controlled real CosyVoice3 synthesis | Step 4A local GPU run | 12/12 cases | Success | Real production path executes mapped rate, arousal, and affect | Globally optimal values or five-class affect accuracy |
| Single-dialogue E2E | Step 6 | 1 dialogue / 8 turns | Pass | Two assigned speakers, clean + telephone, timestamps, metadata, QC | Corpus-scale operations |
| Multi-dialogue pilot | Step 7 | 5 dialogues / 32 turns | 5/5, 32/32 | Sequential batch, speaker reuse, structural QC at small scale | Resume or full-set throughput |
| Small corpus run | Step 9 | 10 dialogues / 54 turns | 10/10, 54/54 | Current-schema corpus path; `--resume` skipped all 10 without worker start | Larger-set stability |
| Larger corpus run | Step 11 | 30 dialogues / 133 turns | 30/30, 133/133 | Full local realistic set under assign → materialize → render | Final upstream schema; perceptual affect/arousal |
| Resume validation | Step 11 second invocation | 30 completed dialogues | skipped=30, rendered=0, retried=0, exit 0, 12.85 s | Dialogue-level resume + fingerprint/QC skip; CosyVoice worker not started | Turn-level resume or crash checkpointing |
| Speaker assignment / materialization | Checked-in scripts + Step 10 tests + Step 11 use | 30 dialogues, seed 5703 | Deterministic JSONL; helper matched Step-9 glue | Repeatable role→`speaker_id`→reference resolution | Formal speaker-similarity scores |
| Full software tests | `uv run --group dev --with pytest pytest` | 454 tests | 454 passed, 1 existing pydub/`audioop` warning | Offline contracts without loading a TTS model | Real-GPU behaviour (covered by the runs above) |

## Evidence boundary

**Rate.** Numeric `slow=0.8` / `normal=1.0` / `fast=1.2` is deterministic.
Real production audio shows directional rate differences. Not shown to be
optimal for every speaker or text.

**Arousal.** Low / medium / high map deterministically to instructions (medium
adds no extra clause). The real synthesis path succeeds. Perceptual arousal
fidelity is not formally established.

**Coarse affect.** `neutral`, `sad`, `anxious`, `angry`, and `warm` execute on
real CosyVoice3. Legacy `distressed` remains compatible. Five-class perceptual
accuracy is not proven.

**Speaker identity.** Assignment is deterministic (seed 5703), roles are stable
within a dialogue, reuse across dialogues is observed, and references are
registry/hash traceable. Formal speaker-similarity evaluation is not claimed.

Exact mapping clauses: [cosyvoice3_control_mapping_v1.md](cosyvoice3_control_mapping_v1.md).

## Larger corpus evidence (Step 11)

Local ignored run: `data/output/corpus_runs/corpus_v0_2_20260909T013249Z/`.

| | |
| --- | --- |
| Dialogues / turns | 30 / 133 (entire tracked `data/input/` set) |
| Result | 30/30 dialogues, 133/133 turns |
| Unique active speakers | 10 |
| Clean / telephone audio | 469.855 s / 469.856 s |
| First-run wall clock | 337.21 s |
| End-to-end wall-clock / generated-clean-audio | **0.718** (not isolated model RTF) |
| Resume | 30 skipped, 0 rendered, 0 retried, exit 0, 12.85 s; worker not initialized |
| QC | All production QC passed |

Two near-full-scale listening flags (`batch_016` turn 4, `batch_023` turn 3).
Not severe clipping failures. Per-dialogue tables live in that run’s
`corpus_summary.md` (gitignored).

## Control mapping (summary)

Implementation is **frozen**. Runtime fidelity status is **provisional**.

| Control | Values | Backend |
| --- | --- | --- |
| Rate | slow / normal / fast | numeric speed 0.8 / 1.0 / 1.2 |
| Arousal | low / medium / high | instruction; medium adds no extra clause |
| Primary affect | neutral, sad, anxious, angry, warm | instruction |
| Legacy affect | distressed | compatibility only |
| Pauses | `pause_before_ms` / `pause_after_ms` | pipeline assembly, not the model |

Mapping name: `cosyvoice3_control_mapping` v1. Details:
[cosyvoice3_control_mapping_v1.md](cosyvoice3_control_mapping_v1.md).

## Batch reliability

`<output>/batch_result.json` is the authoritative batch result.

| Status | Meaning | CLI exit |
| --- | --- | --- |
| `success` | every discovered dialogue succeeded | 0 |
| `partial_failure` | some succeeded, some failed | 1 |
| `failure` | none succeeded | 1 |

A failed dialogue does not stop later dialogues. Successful outputs are kept.
`--resume` is opt-in. Skip-completed requires previous success, matching input
SHA-256, config SHA-256, backend identity, and production QC on required
artifacts. Duplicate current `dialogue_id` values are rejected before any
render. There is no automatic in-process retry or backoff.

## Speaker workflow

Active pool: `data/speaker_pool/vctk_v0.1/` (10 speakers). Seed: **5703**.
Excluded: `spk_002`, `spk_005`. Materialization verifies registry SHA-256 and
rewrites only speaker resolution; text, labels, acoustics, and pauses are
unchanged. Assignment is not part of the `5703tts` CLI.

See [speaker-pool README](../data/speaker_pool/vctk_v0.1/README.md).

## Reproducible demo

Two short canonical dialogues (6 turns total), converted conservatively from
existing realistic `data/input` text. No invented affect/arousal. Requires the
CosyVoice install from the README.

```bash
DEMO=data/output/demo_v0_1
mkdir -p "$DEMO"

uv run python scripts/assign_dialogue_speakers.py \
  --input docs/demo \
  --output "$DEMO/speaker_assignments.jsonl" \
  --summary "$DEMO/speaker_assignment_summary.json" \
  --seed 5703

uv run python scripts/materialize_speaker_assignments.py \
  --input docs/demo \
  --assignments "$DEMO/speaker_assignments.jsonl" \
  --output "$DEMO/render_input" \
  --config-out "$DEMO/config.yaml" \
  --base-config config/config.yaml \
  --manifest "$DEMO/materialization_manifest.json"

uv run 5703tts \
  --input "$DEMO/render_input" \
  --output "$DEMO/production_output" \
  --config "$DEMO/config.yaml" \
  --log-dir "$DEMO/logs"

uv run 5703tts --resume \
  --input "$DEMO/render_input" \
  --output "$DEMO/production_output" \
  --config "$DEMO/config.yaml" \
  --log-dir "$DEMO/logs"
```

What to inspect:

- `production_output/<dialogue_id>/` — `turn_00N.wav`, `*_clean.wav`, `*_telephone.wav`, `*_metadata.json`
- `production_output/batch_result.json` — first run: `action=rendered`, `status=success`; resume: `action=skipped_completed`, `dialogues_rendered=0`
- metadata — requested `acoustic_spec`, `control_resolution`, speaker/reference provenance
- two role voices stay distinct and stable within each dialogue; turn order matches input; pauses are turn-boundary only

This demo does not claim subjective affect quality. Arousal and coarse affect
are null by source.

## Evidence index

Tracked documents:

- [Architecture](ARCHITECTURE.md)
- [Control mapping v1](cosyvoice3_control_mapping_v1.md)
- [Speaker pool v0.1](../data/speaker_pool/vctk_v0.1/README.md)
- [README workflow](../README.md)
- [Code reading guide](CODE_READING_GUIDE.md)
- [Project walkthrough (CN)](PROJECT_WALKTHROUGH_CN.md)

Local gitignored GPU evidence (not in a fresh clone):

| Step | Run directory |
| --- | --- |
| 4A controlled validation | `data/output/experiments/cv3_control_validation_v1_20260908T100933Z/` |
| 6 single-dialogue E2E | `data/output/pilot/single_dialogue_e2e_v1_20260908T112834Z/` |
| 7 five-dialogue pilot | `data/output/pilot/multi_dialogue_pilot_v0_1_20260908T121530Z/` |
| 9 ten-dialogue corpus | `data/output/corpus_runs/corpus_v0_1_20260909T010712Z/` |
| 11 thirty-dialogue corpus | `data/output/corpus_runs/corpus_v0_2_20260909T013249Z/` |

## Upstream dependency

Current renderer input: canonical schema **v0.2**.

The proposed nested upstream acoustic interface is **not formally frozen**.
Do not implement a final nested adapter until that contract is agreed.

When it is frozen, the remaining change should be:

```
upstream final schema  ->  thin adapter  ->  current canonical renderer
```

Do not redesign the production renderer. The adapter should preserve original
upstream semantics; map speaking rate onto current semantic internal rate;
map arousal/coarse affect without fabricating unsupported values; preserve
pause authority as the agreed interface specifies; keep unsupported or
metadata-only fields honest.

## Known limitations

- Final upstream nested interface pending
- Affect/arousal perceptual fidelity remains a separate validation question
- Medium-only `instruct2` generic-prefix edge case
- No within-turn pause positioning from count-only metadata
- No exact hesitation placement from count-only metadata
- No automatic in-process retry/backoff
- No interruption checkpoint beyond dialogue-level `--resume`
- Generated corpus evidence is local and gitignored

## Not current renderer blockers

Evidence does **not** identify these as current renderer blockers:

- speaker assignment
- speaker materialization
- batch rendering
- `batch_result.json` reporting
- dialogue-level resume
- duplicate `dialogue_id` rejection
- clean/telephone generation

## Handoff summary

**Goals**

- Stabilize real CosyVoice3 batch rendering under canonical v0.2
- Make speaker preparation reproducible
- Verify resume and provenance

**Progress made**

- `cosyvoice3_control_mapping` v1 frozen (runtime fidelity provisional)
- 30-dialogue / 133-turn real corpus pass; 469.855 s clean audio
- 454 tests passing (1 existing pydub/`audioop` warning)
- assign → materialize → render workflow checked in
- `--resume` verified: 30 skipped, CosyVoice worker not reloaded

**Blockers**

- Final upstream acoustic schema not yet formally frozen
- Formal perceptual validation of affect/arousal remains separate evaluation work
