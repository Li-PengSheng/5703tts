# Controlled TTS benchmark v0.1

This fixture supports fair, controlled comparison of TTS candidates. Within each comparison group, the text, speaker, crisis label, and every non-target acoustic field remain identical; exactly one requested acoustic variable changes.

The hard controls covered are semantic speaking rate, pause before speech, pause after speech, arousal, and coarse affect. Fine-grained emotion, complex paralinguistic events, speaker-diversity quality, telephone-degradation quality, and downstream crisis-detection usefulness are not evaluated in v0.1.

The machine-readable experimental design is in `benchmark_manifest_v0.1.json`. The exact same canonical schema v0.2 fixture is rendered through CosyVoice and the lightweight local Kokoro baseline so their outputs can be compared under matching requested conditions; it is also intended for a later Fish comparison. Generated audio must not be committed here.

These JSON values describe **requested acoustic conditions**. Their presence does not demonstrate that synthesized speech follows them; acoustic fidelity must be evaluated separately from fixture validity.

## Running a real benchmark

Invoke the runner manually after confirming the selected engine's local model assets:

```bash
# CosyVoice3
uv run python scripts/run_controlled_tts_benchmark.py --config config/config.yaml

# Kokoro
uv run python scripts/run_controlled_tts_benchmark.py --config config/config.kokoro.yaml
```

These commands perform real synthesis; default pytest tests load neither model. Custom `--fixture` and `--manifest` inputs are validated before synthesis to ensure every group changes only its declared target variable while text, speaker, label, and non-target acoustic fields remain identical. One manifest-selected baseline warm-up runs before the eleven timed observations unless `--no-warmup` is supplied. Generated runs, including turn WAVs, the assembled `benchmark_clean.wav`, and `benchmark_results.json`, are written under `data/benchmark/runs/` by default and are gitignored. Use `--output-root` to select another location.

Per-turn RTF is end-to-end `generation_elapsed_sec / audio_duration_sec`. The wall-clock numerator surrounds the production `synthesize_turn` call and can include IPC, frontend work, inference, and file writing; it is not pure GPU inference time. Worker-internal timing remains separately nullable in v0.1.

Kokoro controls semantic rate through its production speed parameter. Pause-before and pause-after are deterministic pipeline assembly controls for both engines, not neural-model controls. Kokoro does not currently consume arousal or coarse affect: those turns are rendered only for full-fixture generation and runtime evidence, and must not be interpreted as Kokoro controllability evidence. CosyVoice's arousal and coarse-affect instruction mappings remain provisional and require empirical verification.

`rate_direction_check` is a non-binding diagnostic, not a pass criterion. All cross-engine comparisons are descriptive and do not declare a winner.
