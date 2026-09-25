# VCTK speaker pool v0.2 — Higgs references

This version freezes **15 Higgs production-approved VCTK reference WAVs**. Ten
retain their active v0.1 `spk_*` identities and copy their v0.1 primary WAVs
byte-for-byte. Five new sources (`p227`, `p294`, `p305`, `p318`, `p343`) were
added as `spk_013` through `spk_017` after offline source listening and Higgs
same-text smoke review. Historical excluded IDs `spk_002` and `spk_005` were
not reused. The v0.1 pool remains immutable historical evidence.

Every frozen reference passed Higgs same-text smoke and human listening with
the same sentence: “I understand. Take your time and tell me what happened when
you are ready.” The five supplemental references were smoked at commit
`6cc2d7cd7178e4b1330c54108fa2497c8c1d10dd`; the ten inherited baseline
references were smoked at commit `c6dfcaf4d084856e3e7715560988173962804c28`.
`approval_evidence.json` records the two cloud log timelines, user-provided
smoke output SHA-256 values, exact reference hashes, and human pass results.
The smoke WAVs and cloud config are absent from this checkout, so the smoke
hashes cannot be recalculated locally and the exact cloud config cannot be
verified here. Smoke outputs are approval evidence only; they are not
production references.

`speaker_registry.json` has one explicit `higgs_reference` with
`approval_status: production_approved` per speaker. The path and SHA-256 refer
to the frozen WAV in `references/`. `active_speakers.json` lists exactly these
15 IDs. The registry intentionally has no `primary_reference`: that field is
for CosyVoice materialization, and the five additions have not undergone a
CosyVoice production approval cycle. This pool is approved for Higgs only.

VCTK source gender, age, accent, and region are descriptive source metadata.
Source age does **not** establish realization of a corpus `age_band`, and VCTK
has no neutral-gender source category. No caller/counsellor partition or
upstream Cxxx/Lxxx identity mapping has been assigned. Production assignment
will be governed by a separate versioned policy and sidecar.

The VCTK 0.92 source is the CSTR VCTK Corpus (CC BY 4.0), DOI
`10.7488/ds/2645`. Reference files were copied without re-encoding.
