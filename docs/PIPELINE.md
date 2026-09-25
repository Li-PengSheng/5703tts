# Pipeline

`5703tts` is the final dialogue-rendering stage of the COMP5703 / CS-28
Controlled-TTS workflow. It accepts final nested JSON/JSONL, selects one
explicit backend, and produces reviewable dialogue audio, provenance, structural
QC, and batch state.

## 1. End-to-end flow

```text
InputRecord
  -> final contract validation
  -> known-issue exclusion policy
  -> speaker sidecar alignment
  -> CanonicalDialogue
  -> backend-specific PreparedDialogue
  -> whole-dialogue preflight
  -> selected backend synthesis
  -> TurnRenderResult
  -> assembly and timestamps
  -> clean WAV + telephone-labelled WAV
  -> metadata + structural QC
  -> Higgs post-render quality assessment + durable quality sidecar
  -> manifest v2
  -> quality-reject operational queue
  -> semantic resume and artifact-integrity checks
```

The renderer accepts a `.json` object or `.jsonl` records. Parsing rejects
duplicate keys, non-standard numbers, non-object records, and blank dialogue
IDs. The final validator accepts the final nested Controlled-TTS contract only;
legacy shapes are not adapted. A malformed JSONL row is isolated as an input
failure so later rows can still be processed.

## 2. Ownership boundaries

| Object | Owns | Does not own |
| --- | --- | --- |
| `InputRecord` | Canonical source snapshot and `record_sha256` | Backend plans or production speaker mutation |
| `CanonicalDialogue` / `CanonicalTurn` | Backend-neutral text, labels, roles, speaker IDs, and normalized controls | Higgs tokens, CosyVoice prompts, or model-native fields |
| `PreparedDialogue` / prepared turn | One selected engine and its validated, defensive-copied execution plan | Replanning during execution |
| `TurnRenderResult` | Prepared-turn identity, speech-only WAV path, synthesis/rate status | Dialogue pauses, metadata interpretation, or perceptual assessment |

The source hash is over canonical JSON content, not container path, JSON
formatting, or JSONL line number. Prepared plans are validated before they are
published and are consumed by execution, metadata, and QC without remapping.
Every turn must pass whole-dialogue preflight before the output directory is
created or the first synthesis request is sent.

## 3. Speaker identity and references

These layers are intentionally distinct:

```text
upstream role:       User / Listener
logical role:        caller / counsellor
scenario speaker:    C001 / L001
production speaker:  spk_001 / spk_008
```

Speaker assignment and backend reference materialization are preprocessing
steps. They write a sidecar and never rewrite source JSON. The renderer reads
speaker and reference information from that sidecar; it does not look up the
registry at runtime.

Higgs uses `higgs_reference.reference_wav` plus its SHA-256. CosyVoice3 uses
`cosyvoice_reference.prompt_wav`, `prompt_text`, and its SHA-256. The contracts
are separate: a CosyVoice primary reference is not automatically an approved
Higgs reference. The frozen v0.2 pool has 15 production-approved Higgs
references; materialization requires approval and matching live hashes. The
frozen v0.2 assignment covers all 1000 canonical dialogues; subsets select its
rows without recomputing speaker assignments.

Internal names `approved_reference` and `approved_speaker_reference` mean that
the prepared render path selected the reference and verified its declared path
and SHA. Formal production approval is a separate registry/sidecar gate,
completed for the frozen v0.2 Higgs pool.

## 4. Backend execution

`tts.engine = higgs | cosyvoice` selects one backend for the complete run.
Higgs3 is the configured default backend; CosyVoice3 is an explicitly selected
secondary backend. There is no automatic fallback.

### Higgs3

The parent owns a standard-library JSON-lines worker. The worker owns and reaps
one SGLang-Omni process group, polls health, sends speech requests, drains
stderr, and reuses the loaded server across turns. Reference conditioning is
sent as `references: [{"audio_path": ...}]`.

Model speed remains `1.0`. Semantic rate is post-processing: `slow` uses FFmpeg
`atempo=0.85`, `normal` has no transform, and `fast` uses `atempo=1.15`.
Higgs arousal and affect use the frozen model-control mapping. `pause_before`
is shared assembly timing; `pause_within` uses the Higgs planner/model-control
path; hesitations use lexical/model-input planning.

### CosyVoice3

CosyVoice runs in its isolated Python environment with a reusable model worker
and `text_frontend=False`. Rate maps to `0.8 / 1.0 / 1.2` for
`slow / normal / fast`. Arousal and affect use provisional instruction
mapping. `pause_before` is shared assembly timing. Positive `pause_within` is
unsupported and fails closed during preflight. Hesitations use the shared
lexical planner.

Runtime execution of either backend does not by itself establish reference
approval, reference-conditioning quality, control fidelity, or perceptual
quality. See [VALIDATION.md](VALIDATION.md) for project-level evidence.

## 5. Controls

| Control | Higgs3 realization | CosyVoice3 realization | Boundary |
| --- | --- | --- | --- |
| `rate` | Fixed model speed `1.0`; FFmpeg `0.85 / 1.0 / 1.15` | Numeric `0.8 / 1.0 / 1.2` | Mapping and runtime execution are not global acoustic validation |
| `arousal` | Frozen model-control mapping | Provisional instruction mapping | Perceptual arousal fidelity remains unvalidated |
| `affect` | Frozen model-control mapping | Provisional instruction mapping | Perceptual affect fidelity remains unvalidated |
| `pause_before` | Shared assembly: `none / short / long` = `0 / 500 / 900 ms` | Same shared assembly timing | Pipeline timing, not model/acoustic validation |
| `pause_within` | Higgs planner/model-control path | Positive values unsupported; preflight fails closed | Higgs token count is not an exact audible-duration claim |
| `hesitations` | Lexical/model-input planning (`0 / 1 / 2`) | Shared lexical planner (`0 / 1 / 2`) | Deterministic planning is not perceptual validation |

The contract also recognizes best-effort `affect_fine`, `volume`,
`flattened_affect`, and `events`. They are requests, not realization guarantees;
CosyVoice does not realize them and Higgs only jointly represents the documented
`affect_fine` rule.

## 6. Outputs and metadata

Each successful dialogue directory contains speech-only `turn_NNN.wav`, a clean
assembled WAV, a telephone-labelled WAV, metadata, and QC results. Assembly
owns `pause_before`, timestamps, turn ordering, and small edge fades. Turn WAVs
contain speech only.

The telephone-labelled file is mono conversion, resampling, high/low-pass
filtering, and level reduction. It is not a codec, packet-loss/noise/room
model, or complete PSTN simulation. Structural QC checks identity, plan/control
agreement, timing, execution status, and WAV readability; it is not perceptual
QA.

Higgs adds post-render termination assessment. Its quality sidecar is durable
authority for the assessed turn-WAV bytes. The per-batch
`quality_rejected.jsonl` is only an operational retry queue; immutable/latest
batch manifests remain execution authority. Controlled retries render fresh
into separate namespaces. The standalone acceptance aggregator combines their
per-dialogue results into a derived delivery index without altering either
authority.

Dialogue metadata records requested controls, the prepared plan, references,
execution, timing, provenance, and backend identity. The field
`runtime_verification: "not_runtime_verified"` means that per-dialogue
metadata does not attest the external runtime environment. It does not mean
the project has never executed a backend on a real GPU; project-level runtime
status belongs in [VALIDATION.md](VALIDATION.md).

## 7. Resume and failure containment

Manifest v2 is the authoritative batch state and is atomically written at
batch completion. A successful resume requires the same semantic render
fingerprint plus valid metadata, WAV artifacts, and live selected-reference
bytes. The fingerprint includes source identity, shared render-affecting
configuration, the selected backend identity, selected speaker/reference
materialization, and exclusion decision. Input location/formatting and
unselected backend state do not affect it.

For Higgs, resume verifies live WAV hashes against quality evidence. A
same-fingerprint quality reject stays terminal under `--resume`; integrity
failure is distinct from ordinary quality rejection. Retry requires an
explicit fresh render, and rejected evidence remains preserved.

Rerender cleanup removes only pipeline-owned direct-child names and fails closed
on managed directories or cleanup errors. Unmanaged files survive. Excluded
records are decided before sidecar/preparation, retain policy provenance in the
manifest, and produce no render output.

## 8. Source map

| Area | Main responsibility |
| --- | --- |
| `src/tts5703/input/` | Strict loading, final validation, exclusion policy |
| `src/tts5703/render_plan.py` | Source + sidecar to canonical/prepared plans |
| `src/tts5703/render_models.py` / `plan_validation.py` | Ownership dataclasses and plan invariants |
| `src/tts5703/tts_engine.py` | Whole-dialogue preflight, explicit dispatch, turn results |
| `src/tts5703/backends/` | Capability mapping, workers, requests, runtime identity |
| `src/tts5703/render/` | Assembly, telephone processing, metadata, structural QC |
| `src/tts5703/pipeline.py` | Single-dialogue ordering and failure containment |
| `src/tts5703/batch.py` / `batch_identity.py` | Manifest v2, resume, cleanup, integrity |
| `src/tts5703/cli.py` | CLI/config selection and batch dispatch |
| `scripts/` | Speaker assignment/materialization and Cloud bootstrap |

For source-level reading, follow `input/` -> `render_plan.py` -> `tts_engine.py`
-> selected `backends/` -> `render/` -> `pipeline.py` -> `batch.py`/`cli.py`.

## 9. Development invariants

- Source records are never rewritten with production speakers or backend plans.
- One run uses one explicitly selected backend; there is no fallback.
- Whole-dialogue preflight completes before the first output/GPU request.
- Turn WAVs are speech-only and `pause_before` is assembly-owned.
- Structural QC is not perceptual QA.
- Backend mapping changes require focused tests and corresponding evidence.
- Offline tests establish software behavior, not GPU execution or acoustic quality.

Run `uv run ruff check .`, `uv run --with pytest pytest -q`, and
`git diff --check` for repository changes. Keep metadata attestation semantics
separate from project-level runtime evidence.
