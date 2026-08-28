# Controlled TTS benchmark v0.1

This fixture supports fair, controlled comparison of TTS candidates. Within each comparison group, the text, speaker, crisis label, and every non-target acoustic field remain identical; exactly one requested acoustic variable changes.

The hard controls covered are semantic speaking rate, pause before speech, pause after speech, arousal, and coarse affect. Fine-grained emotion, complex paralinguistic events, speaker-diversity quality, telephone-degradation quality, and downstream crisis-detection usefulness are not evaluated in v0.1.

The machine-readable experimental design is in `benchmark_manifest_v0.1.json`. The exact same canonical schema v0.2 fixture should later be rendered through CosyVoice and Fish so their outputs can be compared under matching requested conditions. Generated audio must not be committed here.

These JSON values describe **requested acoustic conditions**. Their presence does not demonstrate that synthesized speech follows them; acoustic fidelity must be evaluated separately from fixture validity.

## Running the real CosyVoice3 benchmark

Invoke the runner manually after confirming the local CosyVoice configuration and model assets:

```bash
uv run python scripts/run_controlled_tts_benchmark.py --config config/config.yaml
```

This command performs real synthesis; default pytest tests never load CosyVoice. Custom `--fixture` and `--manifest` inputs are validated before synthesis to ensure every group changes only its declared target variable while text, speaker, label, and non-target acoustic fields remain identical. One manifest-selected baseline warm-up runs before the eleven timed observations unless `--no-warmup` is supplied. Generated runs, including turn WAVs, the assembled `benchmark_clean.wav`, and `benchmark_results.json`, are written under `data/benchmark/runs/` by default and are gitignored. Use `--output-root` to select another location.

Per-turn RTF is end-to-end `generation_elapsed_sec / audio_duration_sec`. The wall-clock numerator surrounds the production `synthesize_turn` call and can include IPC, frontend work, inference, and file writing; it is not pure GPU inference time. Worker-internal timing remains separately nullable in v0.1.

`rate_direction_check` is a non-binding diagnostic, not a pass criterion. Arousal and coarse-affect fidelity still require later listening or prosodic evaluation. Pause-before and pause-after are deterministic assembly controls, not model controls. The same source fixture is intended for a later fair Fish comparison.
