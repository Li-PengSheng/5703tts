# Development

## Environments

The project `.venv` is Python 3.11 and owns the CLI, validation, planning,
assembly, metadata, QC, and tests. CosyVoice3 runs in
`third_party/CosyVoice/.venv` (Python 3.10); Higgs/SGLang runs in
`third_party/sglang-omni/.venv` (Python 3.12). Do not merge dependencies from
the backend environments into the project environment.

## Checks

```bash
uv run ruff check .
uv run --with pytest pytest -q
git diff --check
```

Tests are offline/static. Runtime evidence must be captured separately with the
selected GPU backend and exact repository/config/model identity.

## Module ownership

- `input/`: strict source loading, final contract validation, exclusions.
- `controlled_tts/`: frozen source normalization and Higgs planner contract.
- `speaker_references.py`: selected backend reference parsing and live SHA checks.
- `render_plan.py`: source + sidecar to canonical/prepared ownership boundary.
- `backends/`: selected backend capabilities, workers, requests, and runtime identity.
- `render/`: assembly, telephone-labelled signal processing, metadata, structural QC.
- `pipeline.py`: one-dialogue order and failure containment.
- `batch.py` / `batch_identity.py`: manifest v2, semantic resume, managed cleanup.
- `cli.py` / `config.py`: arguments, direct-child input discovery, one backend selection.
- `scripts/`: speaker assignment/materialization and Cloud bootstrap.

## Architecture invariants

```text
InputRecord -> CanonicalDialogue -> validated PreparedDialogue
-> TurnRenderResult -> assembly/metadata/QC -> manifest/resume
```

Source JSON is never rewritten with production speaker IDs or backend plan data.
`CanonicalDialogue` contains no backend-native fields. Prepared plans are
validated and deep-copied; execution consumes them without remapping. A whole
dialogue passes preflight before the first worker request. One run uses one
explicit backend and never falls back automatically. Turn WAVs are speech-only;
`pause_before` belongs to assembly.

## Changing controls

1. Read `src/tts5703/controlled_tts_v1.json`, the planner, the selected backend
   controls module, and `plan_validation.py`.
2. Decide whether the change is source semantics, backend realization, assembly
   timing, or evidence only.
3. Update the smallest owning module and its focused tests.
4. Update [CONTROLS_AND_BACKENDS.md](CONTROLS_AND_BACKENDS.md) and, when the
   contract changes, [DATA_CONTRACTS.md](DATA_CONTRACTS.md).
5. Keep implementation status separate from control/perceptual validation.

CosyVoice positive `pause_within` must remain fail-closed until a deterministic
supported realization exists. Do not silently reinterpret it or add fallback.

## Adding a backend

Add a config engine value, capability declaration, explicit reference contract,
prepared plan validator/model, backend worker/client, selected dispatch, metadata
identity, batch identity, and tests together. The backend must be explicitly
selected; automatic cross-backend fallback is not part of the current design.

## Evidence versus tests

Offline tests establish software behavior, not GPU execution, speaker identity,
acoustic fidelity, perceptual quality, production-scale throughput, or clinical
usefulness. Runtime evidence must record the actual commit and environment. The
metadata field `runtime_verification: "not_runtime_verified"` remains a
per-dialogue attestation boundary and must not be changed to mirror project-level
operator reports.

## Documentation ownership

- `README.md`: navigation and short project boundary.
- `CURRENT_STATUS.md`: sole current-state source of truth.
- `ARCHITECTURE.md`: design and ownership only.
- `RUNNING.md` / `INSTALLATION_AND_SETUP_CN.md`: operations.
- `DATA_CONTRACTS.md`: persisted and internal shapes.
- `CONTROLS_AND_BACKENDS.md`: cross-backend controls/evidence boundary.
- `CODE_READING_GUIDE.md`: source reading order.
- `docs/evidence/`: specialised/historical evidence; `docs/archive/`: superseded guides.

## Pull-request checklist

- Preserve source/canonical/prepared/result ownership boundaries.
- Keep one selected backend per run and no automatic fallback.
- Add or update focused tests for non-trivial plan/identity changes.
- Update current docs only with claims supported by code or explicitly labelled evidence.
- Keep `runtime_verification` metadata semantics unchanged unless its contract is intentionally redesigned.
- Run ruff, pytest, and `git diff --check`.
