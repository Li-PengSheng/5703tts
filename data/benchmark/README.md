# Controlled TTS benchmark v0.1

This fixture supports fair, controlled comparison of TTS candidates. Within each comparison group, the text, speaker, crisis label, and every non-target acoustic field remain identical; exactly one requested acoustic variable changes.

The hard controls covered are semantic speaking rate, pause before speech, pause after speech, arousal, and coarse affect. Fine-grained emotion, complex paralinguistic events, speaker-diversity quality, telephone-degradation quality, and downstream crisis-detection usefulness are not evaluated in v0.1.

The machine-readable experimental design is in `benchmark_manifest_v0.1.json`. The exact same canonical schema v0.2 fixture should later be rendered through CosyVoice and Fish so their outputs can be compared under matching requested conditions. Generated audio must not be committed here.

These JSON values describe **requested acoustic conditions**. Their presence does not demonstrate that synthesized speech follows them; acoustic fidelity must be evaluated separately from fixture validity.
