# Code reading guide

Read the production path in this order.

1. `src/tts5703/input_records.py` — strict JSON/JSONL loading, duplicate-key and
   non-JSON-number rejection, canonical record hashes.
2. `src/tts5703/final_input.py` — validation of the final upstream dialogue contract.
3. `src/tts5703/cli.py` — the only production batch route, exclusion, manifest v2,
   fingerprinting, and resume.
4. `src/tts5703/final_references.py` — selected-backend reference parsing, path
   resolution, and live SHA verification.
5. `src/tts5703/render_plan.py` — backend-neutral `CanonicalDialogue`, frozen Higgs
   preparation, and cached CosyVoice preparation.
6. `src/tts5703/pipeline.py` — public production `run_dialogue()` orchestration.
7. `src/tts5703/tts_engine.py` and `src/tts5703/backends/` — selected backend
   preflight and one-call-per-turn execution.
8. `src/tts5703/assemble.py`, `metadata.py`, and `qc.py` — common final output path.
9. `src/tts5703/batch_identity.py` — selected-backend semantic resume identity.
10. `scripts/materialize_speaker_assignments.py` — explicit `higgs`, `cosyvoice`, or
    `both` sidecar materialization without source rewriting.

The critical boundary is:

```text
InputRecord -> CanonicalDialogue -> cached selected-backend plan -> TurnRenderResult
```

Higgs is primary; CosyVoice3 is an explicitly selected backup. There is no fallback.
CosyVoice `pause_within > 0` fails closed.

Start tests with:

- `tests/test_final_cli_batch.py`
- `tests/test_batch_identity.py`
- `tests/test_final_dual_backend.py`
- `tests/test_final_orchestration.py`
- `tests/test_materialize_speaker_assignments.py`
- `tests/test_controlled_tts_v1_parity.py`

There is no alternate input validator or renderer dispatch behind the production
command.
