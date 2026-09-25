# Production candidate listening pack

This is a local human-listening candidate pack. It is not a production speaker registry and none of these candidates are production approved.

- `baseline_active`: exact byte copies of the ten existing v0.1 primary reference WAVs; v0.1 files remain unchanged.
- `supplemental`: ten source speakers were initially shortlisted with two different VCTK mic1 excerpts each. A/B are listening labels, not primary/backup decisions. Rejected sources remain in the records, but their WAVs were manually removed.
- `candidates.json`: source metadata, transcripts, source and listening WAV SHA-256 hashes, durations, selection reasons, and technical concerns.
- `review.tsv`: one row per original listening choice, including removed WAVs. First-pass decisions and selected second-pass references are recorded; unassessed detailed listening fields remain blank.

## Human review status

Ten supplemental VCTK sources were initially reviewed. Five (`p298`, `p308`, `p334`, `p362`, `p281`) were rejected in first-pass source listening for insufficient clarity/intelligibility. Second-pass source listening selected these references from the five retained sources:

| Source | Selected choice | Source utterance |
| --- | --- | --- |
| p227 | B | p227_020 |
| p294 | B | p294_017 |
| p305 | B | p305_114 |
| p318 | A | p318_051 |
| p343 | A | p343_012 |

Both A/B WAVs remain for each retained source for audit and possible Higgs review. These are source-listening selections only: none is Higgs approved or production approved, and no caller/counsellor role suitability or assignment was assessed. No Higgs validation has occurred yet.

Supplemental FLACs were converted with FFmpeg to mono 16 kHz PCM16 WAV. No denoising, loudness enhancement, speed or pitch changes, telephone filtering, or added silence were applied. Excerpts passed the existing VCTK candidate-report filters; rankings and transcript inspection do not establish perceptual quality. Age is used only for source-voice diversity and does not realize upstream age bands. Listen for clarity, leading silence, pauses, stability, and suitability before deciding whether to retain any source.
