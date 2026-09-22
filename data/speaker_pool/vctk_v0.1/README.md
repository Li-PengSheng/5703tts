# VCTK Speaker Pool v0.1

This directory holds every speaker-pool artefact for v0.1. The source is CSTR
VCTK Corpus 0.92 under CC BY 4.0. Original VCTK audio and transcript files
remain in the local-only source tree and are never modified. Reported source
paths are relative to that source root.

Speaker identity here is **rendering configuration**. It is not written into
the final upstream source; production `spk_*` identities live in the speaker
sidecar.

## Backend-specific reference approval

`primary_reference.prompt_wav` and `prompt_text` are CosyVoice3-specific: the
selection and refinement lifecycle below used CosyVoice3 zero-shot smoke QA.
The same file is not automatically approved for another backend.

The registry may optionally record an independently reviewed Higgs asset:

```json
"higgs_reference": {
  "reference_wav": "path/to/approved.wav",
  "sha256": "<exact 64-character SHA-256>"
}
```

This means the exact bytes were explicitly approved for Higgs
reference-conditioned synthesis. It carries no transcript, embedding, seed, or
named voice. `scripts/materialize_speaker_assignments.py --backend higgs` (or
`--backend both`) requires this object for every used speaker and verifies the
path and hash before emitting the sidecar.

**No production Higgs references are currently approved.** The existing VCTK
primary WAVs may be tested as candidates in Google Cloud, but the registry must
remain unchanged until real Higgs voice/quality review passes. See
[`docs/VALIDATION.md`](../../../docs/VALIDATION.md).

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
4. **Same-text CosyVoice smoke** - the recorded review evidence synthesizes one
   shared neutral sentence with every speaker reference.
5. **Human review** - listen to the smoke output, refine references where a
   backup was clearly better, and record the final active/rejected decision.
6. **Active speaker pool** - `active_speakers.json` freezes the 10 accepted
   identities. Rejected IDs are kept in provenance; they are never reused as
   active v0.1 voices and never renumbered.
7. **Deterministic dialogue assignment** -
   `scripts/assign_dialogue_speakers.py` maps dialogue roles to active
   `speaker_id` values; the production renderer then resolves them through the
   speaker sidecar.

Stages 1 to 7 are complete for v0.1. Project status reports real Higgs3 GPU
runtime execution, but this pool still has no production-approved
`higgs_reference`; execution is not reference approval or perceptual validation.
See [`docs/VALIDATION.md`](../../../docs/VALIDATION.md).

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

`scripts/assign_dialogue_speakers.py` reads final dialogue JSON/JSONL or a
directory containing those formats and writes:

- `speaker_assignments.jsonl` - one object per dialogue: `dialogue_id`, pool
  version, seed, and `role_assignments` (role → `speaker_id`)
- `speaker_assignment_summary.json` - usage, role, label, and acoustic
  exposure counts for later QA

Assignment is backend-independent and produces only render `speaker_id` values.
The materializer resolves those IDs through `speaker_registry.json` and writes
`higgs_reference`, `cosyvoice_reference`, or both into the production speaker
sidecar according to `--backend`. The renderer reads speaker and reference
information only from that sidecar; it does not resolve the registry at runtime.
Higgs and CosyVoice references are never substituted for one another.

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
uv run python scripts/assign_dialogue_speakers.py \
  --input path/to/final_corpus.jsonl
```

This stage does not synthesise audio.

Resolve assignment JSONL into renderer inputs with
`scripts/materialize_speaker_assignments.py --backend higgs|cosyvoice|both`.
For final input it writes a backend-aware sidecar and never rewrites source
records. It does not assign speakers and does not run TTS.

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

## Speaker-pool smoke evidence

The historical smoke evidence used one shared neutral sentence per speaker.
`speaker_registry.json` preserves the exact VCTK transcript as `prompt_text`; the
sidecar materializer adds the CosyVoice3 zero-shot prefix without rewriting the
registry or source transcripts. Timing results are traceability evidence, not
speaker-quality scores.

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
