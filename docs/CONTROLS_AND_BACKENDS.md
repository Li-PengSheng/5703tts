# Controls and backends

The source contract is shared, but realization is backend-specific. The table
below describes ownership in code; a mapping implementation is not by itself a
perceptual validation result.

## Backend policy

`tts.engine = higgs | cosyvoice` selects one backend for the complete run.
Higgs3 is the configured default backend; CosyVoice3 is an explicitly selected
secondary backend. There is no automatic fallback.

## Required controls

| Control | Higgs3 realization | CosyVoice3 realization | Evidence/status boundary |
| --- | --- | --- | --- |
| `rate` | Model speed is fixed at `1.0`; `slow` uses FFmpeg `atempo=0.85`, `normal` has no transform, `fast` uses `atempo=1.15` | Numeric speed `0.8 / 1.0 / 1.2` for `slow / normal / fast` | Adapter/plan and runtime execution are distinct from global acoustic validation |
| `arousal` | Frozen model-control token mapping, including the documented low+warm combination override | Provisional instruction mapping; medium adds no extra arousal clause | Mapping implementation does not prove perceptual arousal fidelity |
| `affect` | Frozen model-control token mapping | Provisional instruction mapping for neutral, sad, anxious, angry, warm | Mapping implementation does not prove perceptual affect fidelity |
| `pause_before` | Shared assembly timing: `none / short / long` = `0 / 500 / 900 ms` | Same shared assembly timing | Pipeline timing, not a model/acoustic validation claim |
| `pause_within` | Higgs planner/model-control path inserts native pause tokens | Positive values unsupported; whole-dialogue preflight fails closed before synthesis | Higgs native token count is not an exact audible pause-duration claim |
| `hesitations` | Lexical/model-input planning (`0 / 1 / 2`) | Shared lexical planner (`0 / 1 / 2`) | Placement remains deterministic/provisional; it is not perceptual validation |

The source contract also recognizes best-effort `affect_fine`, `volume`,
`flattened_affect`, and `events`. Higgs may mark `affect_fine` as jointly
represented only when the frozen rule matches; the other best-effort fields are
not realized. CosyVoice3 marks best-effort fields unsupported/not realized.

## Higgs3 details

The frozen contract in `src/tts5703/controlled_tts_v1.json` and the adapter in
`src/tts5703/controlled_tts/adapter.py` are the source of truth for tokens,
combination overrides, lexical planning, and postprocess order. The runtime
generation profile keeps model speed at `1.0`; semantic speaking rate is applied
after synthesis with FFmpeg. `pause_before` is not baked into a turn WAV: the
shared assembler adds it before the turn timestamp.

Higgs reference conditioning is a separate contract:
`higgs_reference.reference_wav` plus its SHA-256. A selected reference must be
verified before preparation. Current project status reports real GPU execution,
but formal reference approval and control/perceptual gates remain separate.

## CosyVoice3 details

`src/tts5703/backends/cosyvoice_controls.py` builds the provisional instruction
for `arousal` and `affect`, chooses `zero_shot` when no instruction is needed,
and `instruct2` otherwise. The worker freezes `text_frontend=False`. A
CosyVoice reference is `cosyvoice_reference.prompt_wav` plus exact prompt text
and SHA-256; it is not automatically a Higgs reference.

`pause_within > 0` is represented as unsupported in the prepared plan and is
rejected by `preflight_prepared_turn()` before any worker request. This is a
fail-closed capability boundary, not a fallback path.

## Evidence vocabulary

- `Implemented`: mapping/planner/worker path exists and is guarded structurally.
- `Offline tested`: CPU/static tests and fixtures exercise the path.
- `Runtime verified`: a real GPU runtime executed the selected backend path.
- `Control/perceptual validated`: control fidelity and listening/perceptual evidence are reviewed.
- `Production-scale validated`: representative corpus, resource/recovery, and handoff rehearsal pass.

See [CURRENT_STATUS.md](CURRENT_STATUS.md) for the project-level level assigned
to each backend and [evidence/](evidence/) for specialised historical records.
