# 5703-TTS Higgs Production Runtime, QC & Production Handoff Record

> **Project:** COMP5703 CS-28 — LLM-Generated Audio Dialogues for Crisis Detection in Speech  
> **Workstream:** Rendering Pipeline / Corpus Integration  
> **Backend:** Higgs TTS 3 (`bosonai/higgs-tts-3-4b`) via SGLang-Omni  
> **Production GPU:** NVIDIA L4 (~22.5 GiB VRAM)  
> **Primary repository:** `Li-PengSheng/5703tts`  
> **Updated:** 2026-09-26  
> **Purpose:** Current production handoff for joint execution by the rendering/pipeline owner and the model/acoustic-control teammate  
> **Status:** Runtime, speaker pool, assignment, post-render QC, resume, retry queue, and retry-pack tooling are production-validated. A 30-dialogue / 473-turn pilot reached effective 30/30 acceptance after controlled retry. Full 1000-dialogue rendering has not yet started. The previously identified acoustic-control / speaker-identity issue remains a separate model-mapping concern unless explicitly closed by the model/acoustic-control workstream.

---

## 1. Why this document exists

This document supersedes the earlier 2026-09-25 production log as the **current handoff record**. The earlier record remains useful historical evidence for OOM tuning, cap-hit experiments, and the `high + angry` speaker-identity investigation. fileciteturn168file0

Since then, the production pipeline has advanced materially:

- Higgs audio-quality QC is implemented and fail-closed;
- quality evidence is persisted per dialogue;
- `--resume` semantics have been hardened and production-tested;
- a frozen production speaker pool v0.2 exists;
- a frozen 1000-dialogue speaker assignment v0.2 exists;
- production speaker-reference approval is enforced;
- a `quality_rejected.jsonl` operational retry queue is implemented;
- a deterministic retry-batch builder is implemented;
- the retry workflow has been validated on real rejected outputs;
- a new 30-dialogue / 473-turn production pilot ran for ~63 minutes without OOM or worker crash;
- its two `abnormal_tail` rejects were both recovered by one controlled fresh retry.

The next phase is therefore **joint production execution**, not another round of general runtime experimentation.

---

# 2. Current executive status

## 2.1 Runtime / infrastructure

**PASS — frozen**

```text
max_total_tokens=71680
max_running_requests=1
cuda_graph_max_bs=1
max_new_tokens=1024

temperature=0.8
top_p=0.8
top_k=30
stream=false
speed=1.0
```

Production stack:

```text
Model: bosonai/higgs-tts-3-4b
Serving: SGLang-Omni 0.1.3 / SGLang 0.5.16
GPU: NVIDIA L4 (~22.49 GiB)
Mode: sequential dialogue + sequential turn synthesis
Concurrency: 1
```

Do **not** retune this profile during production unless new evidence shows a runtime defect.

## 2.2 Speaker pool

**PASS — frozen**

Path:

```text
data/speaker_pool/vctk_v0.2/
```

Commit:

```text
4539983c8b8e7ecfe4eba4ff686c31083edeaba1
freeze Higgs production speaker pool v0.2
```

Active production speakers:

```text
spk_001 p248 F23
spk_003 p261 F26
spk_004 p234 F22
spk_006 p283 F24
spk_007 p251 M26
spk_008 p245 M25
spk_009 p237 M22
spk_010 p243 M22
spk_011 p256 M24
spk_012 p252 M22
spk_013 p227 M38
spk_014 p294 F33
spk_015 p305 F19
spk_016 p318 F32
spk_017 p343 F27
```

Pool composition:

```text
15 production speakers
8 female
7 male
ages 19–38
```

All 15 references passed human same-text Higgs listening review and production approval.

Historical rejected IDs remain reserved and must not be reused:

```text
spk_002
spk_005
```

Important limitation: VCTK source ages do not realize corpus `age_band`; `age_band` remains audit/mapping metadata only.

## 2.3 Production reference approval gate

**PASS — enforced**

Commit:

```text
acdb1acfdf6ada23ed036bd9306aaf1ed3f68824
enforce Higgs production reference approval
```

New Higgs sidecar materialization requires exact:

```json
"approval_status": "production_approved"
```

Missing, malformed, unknown, or non-approved references fail closed. Reference SHA-256 is also validated.

## 2.4 Frozen production assignment

**PASS — frozen**

Path:

```text
data/speaker_assignment/v0.2/
```

Commit:

```text
43b5cc880e53e455e526dd82e05966d41dcbde9f
add production speaker assignment policy v0.2
```

Role partition:

```text
CALLER F:
spk_001 spk_003 spk_004 spk_014 spk_015

CALLER M:
spk_008 spk_009 spk_010 spk_011

COUNSELLOR F:
spk_006 spk_016 spk_017

COUNSELLOR M:
spk_007 spk_012 spk_013
```

Properties:

- all 15 production voices used;
- caller/counsellor render sets disjoint;
- full 60 upstream scenario identities mapped once;
- production subsets must select rows from the frozen file;
- production subsets must never recompute assignment.

Canonical corpus SHA-256:

```text
137d74faf441a8e040fe8534b1e17439df541b51f2a3ae2b7e2bc64390027fb0
```

---

# 3. Post-render audio-quality QC

A structurally valid HTTP-200 WAV is not automatically a valid production sample.

Known failure mode:

```text
normal speech
→ voiced content terminates incorrectly
→ long near-silent tail
→ WAV remains readable
→ request still succeeds
```

Current termination policy:

```text
tail_5_drop_db >= 10.0
AND
trailing body-minus-10-dB >= 3.0 s
```

Per-dialogue evidence:

```text
<dialogue_id>_quality.json
```

Quality policy hash:

```text
134cc23da9efd8b1f7e3445232033c369675780512aa3b1707908b2f8adb7b52
```

Key semantics:

- PASS may resume without GPU;
- REJECT is terminal under ordinary `--resume`;
- integrity failure is distinct from quality rejection;
- missing/malformed sidecar or hash mismatch fails closed;
- matching audio can be reassessed offline after a policy change;
- changed render fingerprint requires a fresh render;
- rejected artifacts are preserved.

---

# 4. `--resume` is not retry

`--resume` means **reuse valid existing work**, not stochastic regeneration.

Previous PASS:

```text
same fingerprint
+ valid artifacts
+ valid references
+ valid quality evidence
→ resumed
→ no GPU synthesis
```

Previous same-fingerprint REJECT:

```text
quality_rejected
→ remains rejected
→ WAV preserved
→ no GPU synthesis
```

Production validation on smoke10:

```text
rendered = 0
resumed = 9
failed = 1
elapsed ≈ 0.36 s
GPU synthesis = none
```

Invariant:

```text
resume != retry
```

---

# 5. Controlled quality retry

## 5.1 Design

Retry is intentionally **operator-triggered**, not fully automatic.

```text
production render
    ↓
QC
    ↓
PASS ───────────────→ accepted
    ↓
REJECT
    ↓
quality_rejected.jsonl
    ↓
operator explicitly starts retry
    ↓
build_quality_retry_batch.py
    ↓
fresh retry input + frozen assignment subset
    ↓
materialize speaker sidecar
    ↓
fresh render to NEW output namespace
```

Original rejected artifacts are never overwritten.

## 5.2 Retry queue

Commit:

```text
11d369835c729e024a3bd84d4103a77078091506
add quality rejection retry queue
```

Every completed batch publishes:

```text
<output_root>/quality_rejected.jsonl
```

This is an **operational queue**, not audit authority.

Audit authority remains:

- immutable `batch_result.<attempt>.json`;
- latest `batch_result.json`;
- per-dialogue quality sidecars.

Only genuine retryable quality rejects enter the queue. Integrity failures, input errors, render failures, exclusions, successes and resumed rows do not.

No outstanding rejects:

```text
quality_rejected.jsonl = zero-byte file
```

## 5.3 Retry-batch builder

Commit:

```text
e0770bf0e93f41332f6f7817ba7a5ac7c1a99ec1
add quality retry batch builder
```

Script:

```text
scripts/build_quality_retry_batch.py
```

Inputs:

```text
quality_rejected.jsonl
data/final/corpus_v1_1000.jsonl
data/speaker_assignment/v0.2/speaker_assignments.jsonl
```

Outputs:

```text
input.jsonl
assignments.jsonl
retry_manifest.json
```

Safety properties:

- production `InputRecord.record_sha256` semantics reused;
- canonical record SHA must match retry-queue provenance;
- assignment rows copied, never recomputed;
- queue order preserved;
- duplicate/missing IDs fail closed;
- existing retry attempts not overwritten;
- ordinary staged-publication failures roll back;
- manifest records source/output hashes;
- no TTS/GPU work occurs in the builder.

A real validation showed builder-generated `input.jsonl` and `assignments.jsonl` were byte-identical to the earlier manually prepared retry pack for `corpus_v1_000016`.

---

# 6. Real failure/recovery evidence

## 6.1 smoke10

```text
10 dialogues
150 turns
15/15 production voices
```

First pass:

```text
9 PASS
1 quality_rejected
0 quality_integrity_failed
status = partial_failure
```

Rejected dialogue:

```text
corpus_v1_000016
turn ordinal 10
reason = abnormal_tail
WAV duration = 40.68 s
```

Human listening found roughly 4 s of useful speech followed by a long near-silent tail.

Fresh isolated rerender with the same dialogue, assignment, reference and frozen runtime:

```text
PASS
```

This demonstrated a recoverable stochastic generation failure rather than a broken production reference.

---

# 7. Latest 30-dialogue production pilot

## 7.1 Input

```text
data/review/speaker_v0.2_prod_pilot30/
```

Selection:

```text
canonical dialogues 000000–000029
30 dialogues
473 turns
```

Coverage:

```text
15/15 production voices
caller:      9/9
counsellor:  6/6
```

Hashes:

```text
input.jsonl
98301a04506673c62c95a33cff0881b968d26a9905e0626207d78c09e8740c2e

assignments.jsonl
0a2ba958664e4fc5469215536f41a81f87346fd3b7ed02b1dc2e04b1c0017c7a

speaker_sidecar.json
f40102eafae06f261b5bded6de5a161ae3f3f2614cfdb93f605ccb09241c58f4
```

## 7.2 First-pass result

```text
status = partial_failure

dialogues_total = 30
dialogues_succeeded = 28
dialogues_failed = 2
dialogues_rendered = 28
dialogues_quality_rejected = 2
dialogues_quality_integrity_failed = 0

duration = 3780.519 s
≈ 63.0 min
```

Reference encoder final observed state:

```text
hits = 451
misses = 15
entries = 15
failed = 0
uncacheable = 0
```

Conclusions:

- no observed OOM;
- no worker crash;
- all 15 production references encoded successfully;
- no quality-integrity failure;
- ~1 hour / 473-turn continuous runtime stable.

First-pass acceptance:

```text
28 / 30 = 93.3%
```

Observed reject rate:

```text
2 / 30 = 6.7%
```

This is a small-pilot observation, **not** a population estimate for the full corpus.

## 7.3 Rejected dialogues

`corpus_v1_000013`:

```text
reason = abnormal_tail
ordinal = 1
source_turn_id = 0
duration = 40.68 s
tail_5_drop_db = 31.705...
```

`corpus_v1_000024`:

```text
reason = abnormal_tail
ordinal = 15
source_turn_id = 14
duration = 40.68 s
tail_5_drop_db = 35.516...
```

Both are far beyond the 10 dB threshold and match the recurring 40.68 s cap-hit pattern.

## 7.4 Retry attempt 1

Generated retry pack:

```text
2 dialogues
attempt = 1
corpus_v1_000013 → abnormal_tail
corpus_v1_000024 → abnormal_tail
```

Fresh materialization involved:

```text
spk_010
spk_013
spk_016
```

Fresh retry result:

```text
status = success

dialogues_total = 2
dialogues_succeeded = 2
dialogues_failed = 0
dialogues_rendered = 2
dialogues_quality_rejected = 0
dialogues_quality_integrity_failed = 0

duration = 254.988 s
retry_queue_count = 0
```

Final effective pilot outcome:

```text
30 submitted
28 accepted first pass
2 rejected by QC
2/2 recovered on controlled fresh retry
30/30 effective accepted
```

This is the strongest current end-to-end production evidence.

---

# 8. Cap-hit / 40.68 s interpretation

Historical experiments in the earlier log established:

```text
max_new_tokens = 2048
→ 2041 codec frames
→ ~81.64 s

max_new_tokens = 896
→ 889 codec frames
→ ~35.56 s

max_new_tokens = 1024
→ 1017 codec frames
→ 40.68 s
```

The 1024 cap remains a **safety fuse**, not a semantic fix. fileciteturn168file0

The latest production pilot again produced two rejected WAVs of exactly 40.68 s, strengthening the operational conclusion that this is a recurring model-generation termination mode rather than an isolated corrupted file.

Do not change `max_new_tokens` during production merely to hide this behaviour.

---

# 9. Historical speaker-control / identity investigation

This remains a separate concern from cap-hit QC.

Historical problem case:

```text
corpus_v1_000621 / turn_007
assigned reference: spk_012 / VCTK p252 / male
observed output: clearly female
```

Problem control composition:

```text
<|prosody:expressive_high|><|emotion:anger|>TEXT
```

Earlier targeted experiment summary:

```text
no tag                          3/3 expected male
expressive_high only            3/3 expected male
anger only                      3/3 expected male
expressive_high + anger         3/3 female
same combo, different text      3/3 female
same combo, another male spk    2/3 female, 1/3 male
```

Reverse order reduced but did not eliminate the problem. Anger-only was not globally speaker-safe. Expressive-high-only was the strongest tested candidate in that targeted set:

```text
9/9 preserved expected reference gender
```

Important limitation: human-perceived gender consistency is only a coarse smoke check, not formal speaker verification. fileciteturn168file0

Current handoff rule:

- do not conflate this with runtime stability;
- do not alter frozen runtime to solve it;
- model/acoustic-control teammate should confirm current mapping state before full-corpus production;
- if `high + angry` remains unresolved, resolve/version it explicitly rather than silently changing mapping mid-render;
- speaker identity takes priority over exact realization of optional/fine-grained acoustic controls.

---

# 10. Closed items — do not reopen casually

```text
L4 environment
Higgs model choice
current SGLang serving path
max_total_tokens=71680
max_running_requests=1
cuda_graph_max_bs=1
max_new_tokens=1024 safety fuse
temperature=0.8
top_p=0.8
top_k=30
stream=false
speed=1.0
speaker pool v0.2
production reference approval gate
speaker assignment v0.2
termination QC policy
resume semantics
retry queue semantics
retry-batch builder semantics
```

---

# 11. Open items

## Open A — full 1000-dialogue production

Not yet started.

```text
1000 dialogues
15,934 turns
```

Latest pilot throughput:

```text
3780.519 s / 473 turns
≈ 7.99 s per turn
```

Planning-only extrapolation:

```text
15,934 turns × ~7.99 s
≈ 35.4 GPU hours
```

Recommended execution shape:

```text
10 chunks × 100 dialogues
```

Approximate average per chunk:

```text
~1,593 turns
~3.5 GPU hours
```

Use chunking to bound failures, split work between two people, isolate retry queues, and preserve auditability.

## Open B — acoustic-control mapping sign-off

Before full production, explicitly confirm one of:

```text
1. the historical high+angry issue has already been revised/validated; or
2. it remains open and needs a safety fallback/version; or
3. the affected combination will be excluded/degraded with truthful metadata.
```

## Open C — final acceptance aggregation

Retry outputs live in separate namespaces. Final delivery eventually needs per-dialogue state such as:

```text
accepted_first_pass
accepted_after_retry_01
accepted_after_retry_02
failed_after_retry_limit / manual review
```

Do not overwrite first-pass failure evidence.

---

# 12. Recommended division of work for two people

## Rendering / pipeline owner

Own:

- deterministic production chunk preparation;
- frozen assignment subset selection;
- sidecar materialization;
- cloud production execution;
- batch-result review;
- QC/retry queue handling;
- retry-pack generation;
- attempt provenance;
- final chunk-level acceptance accounting.

Avoid taking on new model/acoustic experiments unless a concrete production failure requires diagnosis.

## Model / acoustic-control teammate

Own:

- confirm current controlled-TTS mapping used by production;
- resolve/sign off the historical `high + angry` identity issue;
- review newly observed speaker/control anomalies;
- decide acceptable control-degradation policy;
- confirm metadata is truthful when a required control is degraded;
- targeted listening of suspect mapping cases;
- advise whether a mapping revision is required before continuation.

## Shared

Agree on:

- exact repo commit for each production chunk;
- frozen runtime config;
- corpus/assignment SHA values;
- retryable-vs-nonretryable failures;
- whether a new failure is model-side or pipeline-side;
- chunk acceptance;
- stop conditions.

---

# 13. Production operating procedure

For each chunk:

```text
1. Record repo commit.
2. Select canonical input rows.
3. Select matching frozen v0.2 assignment rows.
4. Materialize Higgs speaker sidecar.
5. Record SHA-256 provenance.
6. Render chunk once.
7. Read batch_result.json.
8. Read quality_rejected.jsonl.
9. Do NOT rerender successful dialogues.
10. For ordinary retryable quality rejects:
      build retry pack
      materialize retry sidecar
      fresh render into retry_01 namespace
11. Retry again only under an explicit policy.
12. Preserve first-pass and retry evidence.
13. Record final accepted attempt per dialogue.
```

Do not:

```text
overwrite rejected artifacts
use --resume as retry
recompute speaker assignments
change runtime between chunks casually
lower QC threshold to improve success rate
trim a truncated/long-tail WAV and call it valid
rerender a whole successful chunk because a few dialogues failed
silently change acoustic-control mapping mid-production
```

---

# 14. Suggested production directory convention

Preparation:

```text
data/production/v0.2/
├── chunk_000/
├── chunk_001/
...
└── chunk_009/
```

Each chunk:

```text
input.jsonl
assignments.jsonl
speaker_sidecar.json
```

First-pass output:

```text
data/output/production_v0.2/
├── chunk_000/
├── chunk_001/
...
└── chunk_009/
```

Retry namespaces should remain separate, e.g.:

```text
data/production/v0.2_retry/chunk_000_retry01/
data/output/production_v0.2_retry/chunk_000_retry01/
```

---

# 15. Stop conditions during production

Stop/investigate before continuing if any of the following appears:

```text
CUDA OOM recurrence outside known narrow handling
worker/server crash
repeated HTTP failures
reference encode failed > 0
quality_integrity_failed > 0
reference SHA / approval failure
unexpected assignment mismatch
new repeated audio failure mode not covered by QC
systematic speaker identity/control failure
sudden reject-rate spike across many dialogues
```

Ordinary isolated `abnormal_tail` rejects do not require aborting the whole chunk if QC catches them, artifacts remain intact, the queue is generated, and runtime remains healthy.

---

# 16. Current decision log

| Item | Current decision |
|---|---|
| L4 production environment | Keep |
| Higgs model | Keep |
| SGLang serving path | Keep |
| `max_total_tokens=71680` | Frozen |
| `max_running_requests=1` | Frozen |
| `cuda_graph_max_bs=1` | Frozen |
| `max_new_tokens=1024` | Frozen as safety fuse |
| Sampling `0.8 / 0.8 / 30` | Frozen |
| `stream=false` | Frozen |
| Speaker pool v0.2 | Frozen |
| Production reference approval | Required |
| Speaker assignment v0.2 | Frozen |
| Termination QC | Required |
| `--resume` | Reuse only; never retry |
| Quality retry queue | Implemented / validated |
| Retry-batch builder | Implemented / validated |
| Broad automatic rerender | Not approved |
| Controlled fresh retry | Approved operational pattern |
| First-pass reject evidence | Preserve |
| Retry output | New namespace |
| `high + angry` mapping | Historical identity risk; explicit model/acoustic-control sign-off required unless superseded |
| Full 1000-dialogue production | Next execution phase; chunked |

---

# 17. Repository milestones

```text
d503dd8f287b86aebcef66f58960317886fe8c0b
document Higgs audio quality production validation

035790502a5bcb64d34c7f9c396eb6b7ab0433ff
add Higgs production timing instrumentation

7fe6480d376570a5ef4cc63a0b2066b03d62848b
document Higgs production performance validation

acdb1acfdf6ada23ed036bd9306aaf1ed3f68824
enforce Higgs production reference approval

4539983c8b8e7ecfe4eba4ff686c31083edeaba1
freeze Higgs production speaker pool v0.2

43b5cc880e53e455e526dd82e05966d41dcbde9f
add production speaker assignment policy v0.2

11d369835c729e024a3bd84d4103a77078091506
add quality rejection retry queue

e0770bf0e93f41332f6f7817ba7a5ac7c1a99ec1
add quality retry batch builder
```

Latest listed production-tooling commit at handoff time:

```text
e0770bf0e93f41332f6f7817ba7a5ac7c1a99ec1
```

Always record actual HEAD again before generating production chunks.

---

# 18. Minimum evidence a teammate can trust

A teammate joining now does not need to repeat all prior experiments.

```text
Runtime:
stable sequential Higgs on L4

Speaker references:
15/15 production-approved

Assignment:
frozen for all 1000 dialogues

QC:
real abnormal-tail failures detected/rejected

Resume:
validated with zero GPU synthesis

Retry queue:
validated on real rejection

Retry builder:
validated against canonical corpus + frozen assignment

Production pilot:
30 dialogues / 473 turns
28 first-pass PASS
2 abnormal_tail reject
2/2 recovered on fresh retry
effective 30/30 accepted

Runtime stability:
~63 min continuous
15 reference-cache entries
0 reference encode failures
0 quality-integrity failures
no observed OOM / worker crash
```

The teammate's highest-value work is therefore not repeating runtime tuning. It is helping with:

```text
acoustic-control mapping sign-off
speaker/control anomaly review
production chunk review
shared GPU execution/monitoring
final acceptance decisions
```

---

# 19. External technical context

Supporting engineering references only; they are not project audit authority.

1. Higgs TTS 3 model card — autoregressive decoder, 8 audio codebooks, delay pattern, 25 fps:  
   https://huggingface.co/bosonai/higgs-tts-3-4b/blob/main/README.md

2. SGLang-Omni issue #720 — Higgs EOC/max-token overflow reported in **streaming mode**:  
   https://github.com/sgl-project/sglang-omni/issues/720

Caveat:

> Our production config is `stream=false`. The issue is useful analogous evidence that EOC/max-token overflow exists in the ecosystem, but it is not proof that our buffered 40.68 s failures have the identical implementation cause.

---

# 20. Immediate next team meeting outcome

Leave the next meeting with four explicit decisions:

```text
1. Confirm production HEAD and runtime freeze.
2. Confirm whether historical high+angry mapping issue is closed or still open.
3. Split production chunk ownership between the two people.
4. Confirm controlled retry policy and final acceptance bookkeeping.
```

Then proceed:

```text
build deterministic production chunks
→ materialize
→ render chunk_000
→ review
→ continue chunk-by-chunk
```

Do not spend the next session reopening already validated runtime work unless new production evidence demands it.

---

## Maintenance rule

Update this record when any of the following changes:

- runtime profile;
- Higgs/SGLang versions;
- controlled-TTS mapping version;
- speaker pool;
- speaker assignment policy;
- QC policy;
- retry semantics;
- production chunk layout;
- final acceptance policy;
- production commit SHA;
- full-corpus production status.

Historical evidence should not be rewritten after configuration changes. Add a dated decision/result entry so production evolution remains auditable.
