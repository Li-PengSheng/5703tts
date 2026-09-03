# VCTK Speaker Pool v0.1

This directory holds every speaker-pool artefact for v0.1. The source is CSTR
VCTK Corpus 0.92 under CC BY 4.0. Original VCTK audio and transcript files
remain in the local-only source tree and are never modified. Reported source
paths are relative to that source root.

Speaker identity here is **rendering configuration**. It is not written into
the dialogue schema or `acoustic_spec`, and it is not yet used for corpus-scale
production rendering.

## Lifecycle

```
VCTK candidates
→ 12-speaker provisional shortlist
→ CosyVoice3 smoke QA
→ reference refinement
→ 10-speaker active pool
→ deterministic dialogue assignment
→ realistic mini-batch rendering
```

1. **Candidate indexing** - `scripts/prepare_vctk_speaker_pool.py` indexes the
   local corpus into `candidate_report.json` and `candidate_report.tsv`.
2. **Manual shortlist** - a human chooses speakers and their primary/backup
   utterances, recorded in `selected_speakers.json`.
3. **Reference materialisation** - `scripts/materialize_vctk_speaker_pool.py`
   converts the selected recordings into `references/` and writes
   `speaker_registry.json`.
4. **Same-text CosyVoice smoke** - `scripts/run_speaker_pool_smoke.py`
   synthesizes one shared neutral sentence with every speaker reference through
   the existing production synthesis path.
5. **Human review** - listen to the smoke output, refine references where a
   backup was clearly better, and record the final active/rejected decision.
6. **Active speaker pool** - `active_speakers.json` freezes the 10 accepted
   identities. Rejected IDs are kept in provenance; they are never reused as
   active v0.1 voices and never renumbered.
7. **Deterministic dialogue assignment** -
   `scripts/assign_dialogue_speakers.py` maps dialogue roles to active
   `speaker_id` values. Mini-batch rendering comes later and is not implemented
   here.

Stages 1 to 7 are complete for v0.1 except realistic mini-batch rendering.

## Active pool v0.1

Active identities, using the original IDs:

- `spk_001`
- `spk_003`
- `spk_004`
- `spk_006`
- `spk_007`
- `spk_008`
- `spk_009`
- `spk_010`
- `spk_011`
- `spk_012`

Rejected from active v0.1, still present in the historical shortlist, registry,
smoke results, and VCTK provenance:

- `spk_002` - multiple tested references retained undesirable uncontrolled
  pause / leading-silence behaviour.
- `spk_005` - multiple tested references retained undesirable uncontrolled
  pause behaviour.

IDs are never renumbered. `spk_002` and `spk_005` stay `spk_002` and
`spk_005` so earlier artefacts remain traceable.

The 12-row `selected_speakers.json` and `speaker_registry.json` remain the
historical package of every shortlisted identity, including those later
rejected from the active pool.

Speaker demographics copied from VCTK `speaker-info.txt` are descriptive only.
They are not a validation rule and support no population claim.

## Deterministic dialogue assignment

`scripts/assign_dialogue_speakers.py` reads one dialogue JSON file or a
directory of dialogue JSON files and writes:

- `speaker_assignments.jsonl` - one object per dialogue: `dialogue_id`, pool
  version, seed, and `role_assignments` (role → `speaker_id`)
- `speaker_assignment_summary.json` - usage, role, label, and acoustic
  exposure counts for later QA

Assignment is independent of `acoustic_spec`. Prompt WAV and transcript stay
in `speaker_registry.json` and are resolved from `speaker_id` at render time.

Rules:

- Roles in the same dialogue receive different identities.
- Identities recur across dialogues when the batch is large enough.
- The same inputs and seed produce byte-identical JSONL and summary files.
- Filesystem enumeration order does not change the assignment; dialogues are
  sorted by `dialogue_id` then by stable path.
- Crisis labels (`normal` / `alert` / `confirm`) and acoustic fields are **not**
  assignment signals. The summary reports speaker × role × label/acoustic
  exposure so later QA can detect accidental confounding.
- A speaker is not permanently tied to only caller or counsellor when the
  dataset size permits role reuse.

```bash
uv run python scripts/assign_dialogue_speakers.py --input data/input
```

This stage does not synthesise audio.

## Candidate indexing method

- Speakers `p280` and `p315` are excluded from v0.1 because of the documented
  recording/data issues. In particular, p315 has no transcript data.
- `mic1` is used consistently; recordings from other
  microphones are skipped.
- Matched references must be 3.0 to
  8.0 seconds and contain at least
  5 words. A permissive
  4.5-words/second ceiling removes
  transcripts unusually long for their audio duration.
- Ranking favors duration near 5.0
  seconds and word count near 12, with stable
  path-based tie-breakers. These duration and transcript filters are heuristics,
  not perceptual-quality scores.
- Final primary and backup reference selection requires human listening. No
  demographic attributes are inferred from audio; only fields explicitly
  present in `speaker-info.txt` are reported.

## Reference materialisation

Regenerate the reference package from the local corpus with:

```bash
uv run python scripts/materialize_vctk_speaker_pool.py
```

- Every reference is WAV, mono, 16 kHz, signed 16-bit PCM, which is the prompt
  format the CosyVoice zero-shot path expects.
- Only channel, sample-rate, and bit-depth conversion is applied. There is no
  denoising, pitch shifting, speed change, added silence, telephone filtering,
  loudness enhancement, or change of speech content.
- Each written reference is verified for existence, non-zero size, sample rate,
  channel count, 16-bit PCM subtype, and duration preserved within a small
  resampling tolerance. A failed reference fails the whole command rather than
  silently skipping a selected speaker.
- `prompt_text` is read from the VCTK transcript file at materialisation time
  and preserved exactly; no prompt text is hard-coded in Python.

## Speaker identity and provenance

`speaker_id` (for example `spk_001`) is the corpus-facing identifier and encodes
no gender, accent, or region. `source_speaker_id` (for example `p248`) is
retained as provenance, so every reference traces deterministically:

`spk_001` -> `p248` -> `p248_112` -> original VCTK FLAC and transcript ->
materialised reference WAV (recorded with its SHA256).

Rejected identities keep the same chain. Nothing is deleted from the registry
in order to close a gap in the `spk_NNN` sequence.

## Speaker-pool smoke test

The smoke test renders one shared sentence per speaker through
`tts5703.tts_engine.synthesize_turn`, so it reuses the persistent CosyVoice
worker and the production zero-shot path. It maps a temporary in-memory smoke
role to each speaker's reference instead of adding permanent voice-map entries
to `config/config.yaml`. Requested acoustic conditions are deliberately neutral
(normal rate, no pauses, no arousal, affect, emotion, or paralinguistic events),
because the test targets speaker identity, not acoustic control.

`speaker_registry.json` stores the exact VCTK transcript as `prompt_text`, while
the CosyVoice3 zero-shot path expects that transcript prefixed with
`You are a helpful assistant.<|endofprompt|>`. The smoke runner therefore
applies that formatting only to the temporary model-facing config and records
`prompt_format: cosyvoice3_zero_shot` in its report; the registry transcript and
the source VCTK transcripts are never rewritten.

`smoke_results.json` is a smoke-test report only: elapsed time, audio duration,
and real-time factor are recorded for traceability and must not be read as
speaker-quality scores or used to rank speakers. Historical smoke runs are not
rewritten after the active-pool freeze.

## Files

- `candidate_report.json`: complete indexing summary, exclusion/rejection
  counts, source metadata, and up to the configured top K candidates per
  eligible speaker.
- `candidate_report.tsv`: one row per reported candidate for manual review.
- `selected_speakers.json`: the historical 12-speaker shortlist, including
  identities later rejected from the active pool.
- `speaker_registry.json`: generated reference package with provenance,
  transcripts, durations, sample rates, and reference SHA256 values. Still
  contains all 12 shortlisted identities.
- `active_speakers.json`: the frozen 10-speaker active pool and QA reasons for
  identities excluded from active v0.1.
- `references/*.wav`: generated reference audio. Not committed; regenerate it
  locally from VCTK.
- `smoke_runs/<run_id>/`: generated smoke output, results, and review sheet.
  Not committed.
- `speaker_assignments.jsonl`: deterministic role → `speaker_id` mapping per
  dialogue. Does not duplicate prompt audio or transcripts.
- `speaker_assignment_summary.json`: descriptive usage and exposure QA. It is
  not used to retune assignments.

## Licence and attribution

The source corpus is CC BY 4.0, so attribution and provenance must be retained
wherever this material or anything derived from it is shared. The reference WAVs
are derived works of CC BY 4.0 material; they are reproducibly regenerated from
the local corpus and are not committed. The VCTK corpus itself is never
committed to this repository.

## Citation

Yamagishi, Junichi; Veaux, Christophe; MacDonald, Kirsten. (2019).
*CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit
(version 0.92).* University of Edinburgh, CSTR. DOI: 10.7488/ds/2645
