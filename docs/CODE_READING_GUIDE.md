# 5703tts Code Reading Guide

This guide is ordered by questions, not by filename. Read the implementation before the README when they differ.

## Level 1 — Understand external behaviour

1. **`pyproject.toml`** — What command is installed, which Python version is required, and which backends are actually present in the locked main environment?
2. **`config/config.yaml`** — Which engine is selected now? Which role/voice and telephone settings will the default command use? Notice that the opening comment says EdgeTTS while `tts.engine` actually says `cosyvoice`.
3. **`schemas/dialogue_schema.json`** — What is the canonical v0.2 contract, and which acoustic fields are merely accepted rather than required?
4. **`tests/fixtures/dialogue_v0_2.json`** — What does a small canonical dialogue look like in practice?
5. **`src/tts5703/cli.py`** — What does `uv run 5703tts` discover, in what order does it process files, and which failures abort versus remain per-dialogue?

At the end of this level, you should be able to answer: “What goes in, what comes out, and what does the default command select?”

## Level 2 — Understand the core pipeline

1. **`src/tts5703/validate.py`** — How do canonical and versionless legacy JSON converge into `NormalizedDialogue`? Where are turn uniqueness, ordering, and configured speaker checks enforced?
2. **`src/tts5703/pipeline.py`** — What is the exact stage order? When is the output directory created? Which partial outputs survive a failure?
3. **`src/tts5703/engine_capabilities.py`** — What is the difference between `model_control`, `provisional_model_control`, `pipeline_timing`, and `unsupported`?
4. **`src/tts5703/tts_engine.py` lines around rate mappings, preflight, and `synthesize_all_turns()`** — Where does model-independent intent become backend parameters, and why is synthesis sequential?
5. **`src/tts5703/metadata.py`** — Which fields are requested intent, which are configuration declarations, and how are ignored controls exposed?

At the end of this level, trace one `NormalizedTurn` from validation to one turn file and one metadata entry.

## Level 3 — Understand audio processing and alignment

1. **`src/tts5703/assemble.py`** — How do `pause_before_ms`, the previous turn's `pause_after_ms`, fades, and direct concatenation define `start_sec` and `end_sec`?
2. **`tests/test_pipeline_compatibility.py`** — Which concrete timing examples prove there is no overlap?
3. **`src/tts5703/postprocess.py`** — Which telephone effects are actually applied, and which plausible effects (codec/noise/packet loss) are absent?
4. **`src/tts5703/qc.py`** — Which structural properties are checked after output, and which audio-quality properties are not checked?
5. **`tests/test_production_control_metadata.py`** — How is requested intent kept distinct from runtime observation?

At the end of this level, calculate the timestamps of a three-turn dialogue by hand and explain why the last trailing pause is outside the final speech interval.

## Level 4 — Understand each backend

Read focused sections of **`src/tts5703/tts_engine.py`** in this order:

1. `get_engine()`, `turn_audio_extension()`, rate maps — What is shared and what is backend-specific?
2. `synthesize_turn()` EdgeTTS branch — Which request arguments are consumed, and why does this path require the network?
3. Kokoro branch and `_get_kokoro_pipeline()` — How are chunks joined, which sample rate writes the WAV, and how is the model cached?
4. Chatterbox branch — Why is this code experimental despite being selectable?
5. `describe_engine()` — What provenance is produced for each path, and what is missing?

Then read **`config/config.kokoro.yaml`** and **`tests/test_config_validation.py`** to understand the supported local comparison configuration.

At the end of this level, you should be able to distinguish code existence, installed dependency, configured selection, declared capability, and production suitability.

## Level 5 — Understand the CosyVoice boundary

1. **`src/tts5703/tts_engine.py`: `build_cosyvoice_instruction()` and `build_cosyvoice_request()`** — When does the adapter select `zero_shot` versus `instruct2`? Which requested fields never cross the process boundary?
2. **`src/tts5703/tts_engine.py`: `_get_cosyvoice_worker()` through `_cosyvoice_request()`** — How is the worker cached, how is stderr drained, and how are startup/EOF/malformed responses handled?
3. **`src/tts5703/cosyvoice_worker.py`** — What does the Python 3.10 process validate, load, invoke, concatenate, and save?
4. **`tests/test_cosyvoice_controls.py`** — What do deterministic mapping tests prove?
5. **`tests/test_cosyvoice_worker.py`** — How do fake dependencies verify the JSON-lines protocol without a real model?
6. **`tests/test_cosyvoice_worker_lifecycle.py`** — How does the main process terminate and reap a broken worker?
7. **`tests/test_backend_preflight.py`** — Why can `coarse_affect: "anxious"` be schema-valid but CosyVoice-incompatible?

At the end of this level, explain why passing these tests does not prove that an “energetic” instruction creates high-arousal speech.

## Level 6 — Understand tests and failure semantics

1. **`tests/test_validation.py`** — Canonical/legacy compatibility, defaults, ordering, labels, rates, arousal, and role availability.
2. **`tests/test_pipeline_error_paths.py`** — Expected validation/preflight failures versus generic unexpected errors.
3. **`tests/test_engine_capabilities.py`** — Registry completeness and ignored-control semantics.
4. **`tests/test_production_control_metadata.py`** — Provenance fields and expected-versus-runtime sample-rate language.
5. Revisit **`src/tts5703/qc.py`** and note the lack of dedicated QC/postprocess/CLI end-to-end test files.

At the end of this level, list what `243 passed` proves and what still needs real-model, listening, and downstream validation.

## Level 7 — Understand the controlled benchmark

1. **`data/benchmark/README.md`** — What question is benchmark v0.1 designed to answer, and what is explicitly out of scope?
2. **`data/benchmark/benchmark_fixture_v0.1.json`** — How are the eleven observations grouped?
3. **`data/benchmark/benchmark_manifest_v0.1.json`** — Which variable changes in each controlled group?
4. **`scripts/run_controlled_tts_benchmark.py`** — How are confounds rejected, warm-up and RTF measured, partial results persisted, and pauses diagnosed?
5. **`scripts/analyze_controlled_tts_benchmark.py`** — Which waveform descriptors are measured, why F0 is absent, and why conclusions remain descriptive?
6. **`tests/test_benchmark_fixture.py`**, **`tests/test_benchmark_runner.py`**, **`tests/test_benchmark_acoustic_analysis.py`** — Which design assumptions and failure paths are executable specifications?

At the end of this level, explain why keeping text, speaker, label, and all non-target fields constant is necessary for attributing a difference to one requested control.

## Five-file fast path

If you only have an hour, read:

1. `schemas/dialogue_schema.json`
2. `src/tts5703/validate.py`
3. `src/tts5703/pipeline.py`
4. `src/tts5703/tts_engine.py`
5. `src/tts5703/assemble.py`

Then use `tests/test_production_control_metadata.py` as the first follow-up because it captures the project's most important semantic distinction: a requested control is not an observed acoustic fact.
