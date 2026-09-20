# Code reading guide

Read the production path in this order.

1. `src/tts5703/input/records.py` — strict JSON/JSONL loading, duplicate-key and
   non-JSON-number rejection, canonical record hashes.
2. `src/tts5703/input/contract.py` — validation of the final upstream dialogue contract.
3. `src/tts5703/speaker_references.py` — selected-backend reference parsing, path
   resolution, and live SHA verification.
4. `src/tts5703/render_models.py` — canonical and prepared rendering data models.
5. `src/tts5703/render_plan.py` — canonical construction, frozen Higgs preparation,
   and cached CosyVoice preparation.
6. `src/tts5703/tts_engine.py` and `src/tts5703/backends/` — selected backend
   preflight and one-call-per-turn execution.
7. `src/tts5703/render/` — common assembly, postprocessing, metadata, and QC output path.
8. `src/tts5703/pipeline.py` — public production `run_dialogue()` orchestration.
9. `src/tts5703/batch_identity.py` and `src/tts5703/batch.py` — semantic resume
   identity, exclusion, manifest v2, and production batch orchestration.
10. `src/tts5703/cli.py` — argument parsing, config and input loading, then batch
    dispatch.
11. `scripts/materialize_speaker_assignments.py` — explicit `higgs`, `cosyvoice`, or
    `both` sidecar materialization without source rewriting.

The critical boundary is:

```text
InputRecord -> CanonicalDialogue -> cached selected-backend plan -> TurnRenderResult
```

Higgs is primary; CosyVoice3 is an explicitly selected backup. There is no fallback.
CosyVoice `pause_within > 0` fails closed.

Start tests with:

- `tests/test_cli_batch.py`
- `tests/test_batch_identity.py`
- `tests/test_dual_backend.py`
- `tests/test_orchestration.py`
- `tests/test_materialize_speaker_assignments.py`
- `tests/test_controlled_tts_v1_parity.py`

There is no alternate input validator or renderer dispatch behind the production
command.
