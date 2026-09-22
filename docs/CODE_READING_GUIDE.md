# Code reading guide

Read the production path in this order. Each stop answers one question, states its ownership boundary, and points to the next file. For current evidence levels see [CURRENT_STATUS.md](CURRENT_STATUS.md); for persisted shapes see [DATA_CONTRACTS.md](DATA_CONTRACTS.md).

## 1. Input containers and source identity

Read `src/tts5703/input/records.py`.

- Question: How do JSON/JSONL bytes become stable record identities and isolated failures?
- Owns: strict parsing, duplicate-key/NaN/Infinity rejection, canonical content SHA, record-level JSONL failure retention.
- Does not own: final dialogue schema or acoustic-control domains.
- Next: `input/contract.py` and `controlled_tts/schema.py`.

## 2. Final contract and normalization

Read `src/tts5703/input/contract.py`, then `src/tts5703/controlled_tts/schema.py`.

- Question: What exact source shape and controls are accepted?
- Owns: final nested validation; User/Listener to caller/counsellor normalization; scenario speaker lookup; required and best-effort requests.
- Does not own: production `spk_*` assignment or backend-native controls.
- Next: `input/exclusions.py`, then `speaker_references.py`.

## 3. Exclusion policy

Read `src/tts5703/input/exclusions.py`.

- Question: How are known issues excluded without hard-coded dialogue IDs?
- Owns: strict external policy, unique IDs, reasons/provenance, policy SHA.
- Does not own: rendering, speaker lookup, or source mutation.
- Next: `speaker_references.py`.

## 4. Selected references

Read `src/tts5703/speaker_references.py`.

- Question: How does the selected backend get its exact reference and verify live bytes?
- Owns: distinct Higgs/CosyVoice reference contracts, declared/resolved paths, SHA verification.
- Does not own: assignment generation or control planning.
- Next: `render_models.py`.

## 5. Ownership data models

Read `src/tts5703/render_models.py`.

- Question: What belongs to a source turn, canonical turn, prepared turn, and turn result?
- Owns: dataclass boundaries, prepared-constructor validation, defensive plan copies, derived execution properties.
- Does not own: transformation logic or synthesis.
- Next: `render_plan.py`, then `plan_validation.py`.

## 6. Canonicalization and preparation

Read `src/tts5703/render_plan.py` and `src/tts5703/plan_validation.py`.

- Question: How do source + sidecar become one frozen backend plan?
- Owns: identity alignment, production speaker introduction, canonical controls, selected live references, Higgs/CosyVoice plan construction and plan invariants.
- Does not own: worker startup, WAV creation, or perceptual QA.
- Next: `tts_engine.py`.

## 7. Whole-dialogue execution boundary

Read `src/tts5703/tts_engine.py`.

- Question: Why must every turn pass before the first synthesis request, and how is backend dispatch chosen?
- Owns: complete-dialogue preflight, explicit dispatch, sequential turn execution, `TurnRenderResult` creation.
- Does not own: backend process details or batch resume.
- Next: the selected backend files.

Backend dispatch:

```text
preflight_prepared_dialogue / synthesize_prepared_turns
  +-- engine=higgs --> backends/higgs.py --> higgs_worker.py --> SGLang HTTP
  `-- engine=cosyvoice --> backends/cosyvoice.py
                         --> cosyvoice_client.py --> cosyvoice_worker.py
```

## 8. Higgs backend

Read `src/tts5703/backends/higgs_controls.py`, `backends/higgs.py`, `backends/higgs_worker.py`, then `backends/info.py`.

- Question: How are frozen control identity, worker/server ownership, reference conditioning, fatality, and FFmpeg rate processing implemented?
- Owns: contract pin; cached worker keyed by runtime settings; JSON-lines protocol; stderr draining; SGLang process group, health/speech HTTP; atomic raw WAV; atempo; backend identity.
- Does not own: source normalization, dialogue pause assembly, or evidence that conditioning sounds correct.
- Next: `render/assemble.py`.

## 9. CosyVoice backend

Read `src/tts5703/backends/cosyvoice_controls.py`, `backends/capabilities.py`, `backends/cosyvoice.py`, `backends/cosyvoice_client.py`, `backends/cosyvoice_worker.py`, then `backends/info.py`.

- Question: How does the explicit backup backend isolate its environment and execute cached plans?
- Owns: provisional speed/instruction mapping, unsupported pause-within declaration, prompt checks, reusable worker/model, `text_frontend=False`, selected-backend identity.
- Does not own: automatic fallback or perceptual fidelity claims.
- Next: `render/assemble.py`.

## 10. Audio, provenance, and QC

Read `src/tts5703/render/assemble.py`, `postprocess.py`, `metadata.py`, and `qc.py`.

- Question: How do speech-only turn WAVs become clean/telephone dialogue and reviewable provenance?
- Owns: exact turn matching, pause insertion, timestamps, edge fades, telephone signal processing, requested/planned/executed metadata, structural/control-integrity QA.
- Does not own: synthesis mapping, codec/PSTN simulation, perceptual/clinical QA.
- Next: `pipeline.py`.

## 11. Single-dialogue orchestration

Read `src/tts5703/pipeline.py`.

- Question: What is the exact production order and where are exceptions contained?
- Owns: validate -> prepare -> whole-dialogue preflight -> output directory -> synthesize -> assemble -> clean/telephone -> metadata -> QC -> `PipelineResult`.
- Does not own: cross-record decisions, stale cleanup, or manifest writing.
- Next: `batch_identity.py`, then `batch.py`.

## 12. Semantic resume and batch state

Read `src/tts5703/batch_identity.py`, then `src/tts5703/batch.py`.

- Question: When is previous output semantically identical and still trustworthy?
- Owns: selected-backend fingerprint components, manifest v2, exclusion ordering, path security, artifact/reference integrity, managed cleanup, per-record action/status, final batch write.
- Does not own: control remapping during resume or arbitrary recursive cleanup.
- Next: `cli.py` and `config.py`.

## 13. CLI and config

Read `src/tts5703/config.py`, then `src/tts5703/cli.py`.

- Question: How is one backend selected and how are input containers dispatched?
- Owns: selected runtime validation, argument parsing, direct-child CLI discovery, duplicate dialogue-ID rejection, `run_batch` dispatch.
- Does not own: fallback, recursive CLI discovery, or rendering logic.
- Next: sidecar materialization script.

## 14. Speaker materialization

Read `scripts/materialize_speaker_assignments.py`.

- Question: How are source roles assigned production speakers without rewriting source records?
- Owns: four preprocessing inputs, active/registry/reference validation, explicit `higgs`/`cosyvoice`/`both` target, sidecar output.
- Does not own: runtime reference lookup from registry or automatic Higgs approval.
- Next: relevant tests and runtime runbooks.

## Call graph

```text
cli.main
  -> load_config / read_input_records / duplicate-ID check
  -> run_batch
      -> semantic fingerprint / resume-integrity decision / stale cleanup
      -> run_dialogue
          -> validate_dialogue
          -> prepare_dialogue
              -> canonicalize_dialogue
              -> Higgs or CosyVoice prepared-plan construction
          -> preflight_prepared_dialogue
              -> selected backend preflight_prepared_turn
          -> synthesize_prepared_turns
              -> selected backend synthesize_prepared_turn (sequential)
          -> assemble_prepared_dialogue
          -> apply_telephone_effect
          -> build_metadata / write_metadata
          -> run_qc
      -> write manifest v2
```

## Tests to anchor the reading

- `tests/test_input_records.py` and `tests/test_input_contract.py`
- `tests/test_render_plan.py` and `tests/test_orchestration.py`
- `tests/test_dual_backend.py`
- `tests/test_batch_identity.py` and `tests/test_cli_batch.py`
- `tests/test_materialize_speaker_assignments.py`
- `tests/test_controlled_tts_v1_parity.py`

The critical invariant is:

```text
InputRecord -> CanonicalDialogue -> validated cached PreparedDialogue
            -> TurnRenderResult -> assembly/metadata/QC -> manifest/resume
```

There is no alternate production schema adapter, hidden renderer dispatch, or automatic backend fallback.
