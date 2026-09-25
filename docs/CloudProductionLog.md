# 5703-TTS Higgs Production Runtime & Audio Quality Record

> **Project:** COMP5703 CS-28 — LLM-Generated Audio Dialogues for Crisis Detection in Speech  
> **Workstream:** Rendering Pipeline / Corpus Integration  
> **Backend:** Higgs TTS 3 (`bosonai/higgs-tts-3-4b`) via SGLang-Omni  
> **Production GPU:** NVIDIA L4 (~22.5 GiB VRAM)  
> **Record date:** 2026-09-25  
> **Status:** Runtime profile frozen; main-corpus rendering remains on hold pending audio-quality / mapping safeguards.

---

## 1. Purpose of this record

This document records the productionisation work performed after Higgs TTS became runnable on the cloud L4 environment.

It is intended to be maintained as the project-facing engineering record for:

- cloud/runtime setup;
- Higgs/SGLang production hardening;
- CUDA OOM investigation;
- retry policy;
- KV/cache/CUDA graph tuning;
- `max_new_tokens` experiments;
- pilot30 production validation;
- performance observations;
- cap-hit / runaway generation behaviour;
- speaker identity instability;
- targeted acoustic-control experiments;
- current frozen configuration;
- unresolved production gates;
- discussion points for alignment with the model/acoustic-control workstream.

The document deliberately separates:

1. **confirmed facts**;
2. **experimental evidence**;
3. **current engineering decisions**;
4. **candidate changes not yet implemented**.

---

# 2. Current high-level status

## 2.1 Runtime / infrastructure

**PASS**

The current serving/runtime configuration is stable enough to freeze:

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

Observed final fresh pilot behaviour:

```text
30/30 dialogues rendered
0 failed
0 resume
0 observed CUDA OOM
0 retry
0 HTTP 500
0 server death
```

Runtime/server stability is therefore no longer the main blocker.

---

## 2.2 End-to-end corpus quality

**NOT YET PRODUCTION-PASS**

Two independent audio-quality failure modes were identified:

1. **runaway/cap-hit generation**
   - reproducible on `corpus_v1_000621 / turn_012.wav`;
   - generated until the configured token ceiling;
   - final 1024 profile produced 1017 codec frames / ~40.68 s;
   - audible long trailing silence;
   - must not silently count as a valid production turn.

2. **speaker identity instability under control composition**
   - reproduced on `corpus_v1_000621 / turn_007.wav`;
   - requested/reference speaker is male `spk_012`;
   - generated output became clearly female under the `expressive_high + anger` control combination;
   - targeted experiments show this is not caused by the runtime profile itself.

Therefore:

> **30/30 rendered is a transport/render success metric, not a quality-valid corpus success metric.**

Main-corpus generation should remain **HOLD** until the mapping/QC policy is agreed.

---

# 3. Production cloud environment

## 3.1 Hardware

Current production test environment:

```text
GPU: NVIDIA L4
VRAM: approximately 22.5 GiB
```

The rendering pipeline is intentionally sequential:

```text
dialogues: sequential
turns: sequential
concurrency: 1
```

No worker pool / multi-request throughput scaling is currently required for correctness.

---

## 3.2 Higgs runtime stack

Production backend:

```text
Model: bosonai/higgs-tts-3-4b
Serving: SGLang-Omni 0.1.3 / SGLang 0.5.16
Endpoint: /v1/audio/speech
```

The application-side Higgs worker owns the SGLang process and:

- starts the server;
- waits for readiness;
- reuses one loaded server;
- sends reference-conditioned `/v1/audio/speech` requests;
- validates returned WAV files;
- owns shutdown/reaping;
- keeps the main process isolated from SGLang lifecycle failures.

---

## 3.3 Cloud bootstrap update

The cloud setup/bootstrap was updated to include:

```text
ninja-build
```

and readiness checks require `ninja`.

The intention was explicitly **not** to upgrade unrelated CUDA/SGLang/model dependencies during production hardening.

---

# 4. Initial production failure: CUDA OOM

The first realistic pilot exposed CUDA out-of-memory failures.

Initial generation profile included:

```text
max_new_tokens=2048
```

At the earlier runtime configuration, pilot30 completed only after resume:

```text
first pass:
25 succeeded
5 CUDA OOM

resume:
remaining 5 completed
final total: 30/30
```

This demonstrated that offline/mock test success did not establish production GPU reliability.

---

# 5. OOM retry policy

A deliberately narrow retry mechanism was added.

## 5.1 Retry condition

Retry **once only** when:

```text
HTTP status = 500
AND
response body explicitly indicates CUDA out of memory
```

Maximum:

```text
2 total attempts
```

Same loaded server is reused if it is still healthy.

---

## 5.2 Explicit non-retry cases

The pipeline does **not** automatically retry:

- generic HTTP 500;
- HTTP 4xx;
- timeout;
- `URLError` / transport failures;
- server death;
- malformed WAV;
- arbitrary synthesis failure.

Reason:

> Retry behaviour must remain narrow and auditable rather than turning every synthesis problem into an uncontrolled stochastic rerender.

---

## 5.3 Resume semantics

Existing batch behaviour remains:

- attempt evidence remains immutable;
- latest `batch_result` represents current result;
- resume remains dialogue-level;
- no partial-turn/partial-dialogue resume was introduced.

---

# 6. Runtime memory tuning

The goal was to reduce OOM risk without changing model behaviour unnecessarily.

---

## 6.1 `max_total_tokens`

Selected:

```text
max_total_tokens=71680
```

This reduced the KV allocation and recovered approximately ~1 GiB class memory compared with the larger baseline.

The purpose was **VRAM headroom**, not higher token/s throughput.

---

## 6.2 `max_running_requests`

Selected:

```text
max_running_requests=1
```

Because the production pipeline already sends one request at a time, allowing a large number of concurrent running requests provides no useful benefit for the current workload.

---

## 6.3 `cuda_graph_max_bs`

Selected:

```text
cuda_graph_max_bs=1
```

Because effective request batch size is one.

Observed CUDA graph memory reduced from approximately:

```text
~0.42 GiB
```

to approximately:

```text
~0.03 GiB
```

recovering roughly ~398 MiB class VRAM.

No meaningful single-request throughput regression was observed at the time of this change.

---

# 7. `max_new_tokens` investigation

The difficult turn:

```text
dialogue: corpus_v1_000621
turn: turn_012.wav
```

repeatedly ran to the configured ceiling.

This became the strongest evidence of a model termination/runaway failure.

---

## 7.1 2048 profile

Configuration:

```text
max_new_tokens=2048
```

Observed abnormal turn:

```text
codec frames: 2041
expected ceiling relationship: 2048 - 7 = 2041
WAV duration: ~81.64 s
```

This is an exact cap signature.

---

## 7.2 896 profile

Configuration:

```text
max_new_tokens=896
```

Same abnormal turn:

```text
codec frames: 889
expected ceiling relationship: 896 - 7 = 889
WAV duration: ~35.56 s
```

Again an exact cap signature.

---

## 7.3 Final 1024 profile

Selected:

```text
max_new_tokens=1024
```

Same turn:

```text
codec frames: 1017
expected ceiling relationship: 1024 - 7 = 1017
WAV duration: 40.680 s
```

Again exact cap behaviour.

This establishes:

> The abnormal turn does not terminate normally. Changing the ceiling only changes how long the runaway generation is allowed to continue.

Therefore increasing the token cap is **not** considered a fix.

---

# 8. Why 1024 was selected

`1024` is currently treated as a **safety fuse**, not as the solution to the termination problem.

Benefits:

- prevents the 80+ second 2048-style runaway;
- leaves more bounded output size;
- materially reduces worst-case GPU exposure;
- preserves enough room for normal turns observed in the pilot.

Rejected idea:

```text
raise to 1152 / 1280 / 2048
```

Reason:

> This would allow known abnormal generation to continue for longer and would move the system back toward OOM risk.

---

# 9. Pilot results

## 9.1 Fresh 71680 / 1 / 1 / 2048

Observed:

```text
30/30
duration: 3860.652 s
0 OOM
0 retry
0 HTTP 500
minimum free VRAM observed: ~253 MiB
```

Worst runaway turn:

```text
corpus_v1_000621 / turn_012
2041 codec frames
~81.64 s
```

---

## 9.2 Fresh 71680 / 1 / 1 / 896

Observed:

```text
30/30
duration: 3827.522 s
minimum free VRAM observed: ~1103 MiB
```

Same runaway turn:

```text
889 codec frames
~35.56 s
```

---

## 9.3 Final fresh 71680 / 1 / 1 / 1024

Observed:

```text
30/30 dialogues
30 rendered
0 failed
0 resumed
duration: 3982.824 s
```

Final-only reliability log checks found:

```text
0 CUDA OOM
0 retry
0 HTTP 500
0 server death
```

Maximum codec frame event:

```text
1017
```

Exactly one event reached this cap-like value.

Mapped turn:

```text
corpus_v1_000621 / turn_012.wav
```

Duration:

```text
40.680 s
```

---

# 10. Performance interpretation

Comparison:

```text
2048 fresh pilot: 3860.652 s
1024 final pilot: 3982.824 s
difference: +122.172 s
relative increase: ~3.16%
```

This is below the previously used:

```text
>5% unexplained throughput regression
```

gate.

Important interpretation:

The runtime parameters changed primarily for memory safety.

They do not inherently make one sequential request faster:

```text
max_total_tokens      → memory/KV capacity control
max_running_requests  → concurrency capacity
cuda_graph_max_bs     → graph coverage / graph memory
max_new_tokens        → output ceiling
```

The current generation profile remains stochastic:

```text
temperature=0.8
top_p=0.8
top_k=30
```

No fixed seed is used.

Therefore full-batch wall-clock time contains variation from:

- generated token count;
- output duration;
- turn-level stochastic termination;
- abnormal generation.

For future performance comparison, prefer collecting:

```text
total generated tokens
generation token/s
total rendered audio duration
wall-clock batch time
```

rather than interpreting wall-clock duration alone.

Current conclusion:

> No runtime rollback is justified by the observed +3.16% wall-clock variation.

---

# 11. Audio quality issue 1: cap-hit + trailing silence

Human listening of:

```text
corpus_v1_000621 / turn_012.wav
```

found:

- long trailing silence;
- output is perceptually invalid;
- file itself is structurally valid WAV;
- HTTP status is 200;
- current pipeline therefore records it as success.

This exposes the following correctness gap:

```text
abnormal model generation
→ HTTP 200
→ readable WAV
→ pipeline success
```

That is not sufficient for production corpus quality.

---

# 12. Audio quality issue 2: speaker identity failure

A second independent problem was found in:

```text
corpus_v1_000621 / turn_007.wav
```

Source mapping:

```text
source_turn_id: 6
ordinal: 7
logical_role: caller
render_speaker_id: spk_012
reference: spk_012_primary.wav
```

Speaker registry provenance:

```text
spk_012
→ VCTK p252
→ male
→ Scottish
```

Human listening found:

> `turn_007.wav` was rendered as a clearly female voice even though the assigned/reference speaker is male.

This is a production correctness failure, not an intended dialogue speaker switch.

---

# 13. Historical reproduction of `turn_007`

The same turn was checked across previous pilot outputs.

Observed:

```text
higgs_pilot30
→ female
→ no long trailing silence

higgs_pilot30_kv71680_bs1
→ female
→ trailing/blank behaviour present

higgs_pilot30_kv71680_bs1_cap896
→ female
→ trailing/blank behaviour present

final_1024
→ female
```

The identity problem is therefore not specific to the final `1024` cap.

This is important because it largely separates:

```text
runtime memory tuning
```

from:

```text
speaker/control conditioning quality
```

---

# 14. Problem turn control composition

The problematic source turn text:

```text
i hate that i can't sleep. i just feel so tired and sad and like my brain is
stuck on repeat. i don't think i'll be able to live like this much longer.
```

Current controlled-TTS model input:

```text
<|prosody:expressive_high|><|emotion:anger|>TEXT
```

The current Controlled TTS v1 mapping independently defines:

```text
angry
→ <|emotion:anger|>
→ fully_realized

high arousal
→ <|prosody:expressive_high|>
→ context_dependent_proxy
```

Before this production investigation, their joint use was not backed by explicit combination-level validation.

---

# 15. Targeted A/B speaker-identity investigation

To isolate the cause, a direct Higgs worker probe was run using the frozen production runtime.

The tests reused:

```text
71680 / 1 / 1 / 1024
temperature=0.8
top_p=0.8
top_k=30
```

No repository production mapping was changed during these experiments.

---

## 15.1 A–F experiment

### A — no tags, spk012, original text

```text
3/3 male
```

### B — `expressive_high` only, spk012, original text

```text
3/3 male
```

### C — `anger` only, spk012, original text

```text
3/3 male
```

### D — `expressive_high + anger`, spk012, original text

```text
3/3 female
```

### E — `expressive_high + anger`, spk012, different text

```text
3/3 female
```

### F — `expressive_high + anger`, another male speaker

Observed:

```text
2/3 female
1/3 male
```

### Initial conclusion

The failure is not explained by:

- `expressive_high` alone;
- `anger` alone;
- original text alone;
- `spk_012` alone.

The **combined control condition** creates a strong identity-instability risk.

---

# 16. Tag-order experiment

The combined tag order was reversed.

Production ordering:

```text
<|prosody:expressive_high|><|emotion:anger|>
```

Experimental reverse ordering:

```text
<|emotion:anger|><|prosody:expressive_high|>
```

Results:

### G — reverse order, spk012, original problem text

```text
2/3 female
1/3 male
```

### H — reverse order, spk012, other text

```text
3/3 male
```

### I — reverse order, another male speaker

```text
3/3 male
```

### Conclusion

Changing tag order reduces the failure in some contexts but **does not eliminate it**.

Therefore:

> Tag reversal is not safe enough to be considered a production fix.

The behaviour is interaction-dependent across:

```text
speaker × text × acoustic-control composition
```

---

# 17. Anger-only fallback experiment

An anger-only fallback was then tested more broadly.

Existing earlier result:

```text
spk012 + original problem text + anger-only
→ 3/3 male
```

Additional experiment:

### J — spk012 + other text + anger-only

```text
3/3 male
```

### K — another male speaker + original problem text + anger-only

Observed:

```text
1/3 male
2/3 female
```

### L — female reference speaker + original text + anger-only

Observed:

```text
3/3 female
```

### Conclusion

`anger-only` cannot be considered a globally speaker-safe fallback.

It worked for `spk_012`, but failed cross-speaker on another male reference.

Therefore the following candidate change was **rejected for now**:

```text
high + angry
→ anger-only
```

---

# 18. Expressive-high-only cross-speaker experiment

A final cross-speaker test was run for `expressive_high` alone, with no-tag controls.

### M — male speaker + original text + expressive_high

```text
3/3 male
```

### N — female speaker + original text + expressive_high

```text
3/3 female
```

### O — male speaker + original text + no tag

```text
3/3 male
```

### P — female speaker + original text + no tag

```text
3/3 female
```

Together with the earlier:

```text
spk012 + expressive_high
→ 3/3 male
```

current test evidence for `expressive_high-only` is:

```text
9/9 outputs preserved expected reference gender
```

within this targeted experiment set.

---

# 19. What the experiments currently establish

## Strong evidence

The existing production composition:

```text
<|prosody:expressive_high|><|emotion:anger|>
```

is unsafe for reliable speaker identity.

It should not be treated as a validated production combination.

---

## Also established

The following are **not** sufficient fixes:

```text
increase max_new_tokens
change KV capacity
change CUDA graph batch size
reverse tag order
use anger-only globally
```

---

## Current best candidate

The strongest currently tested fallback is:

```text
high + angry request
→ synthesize using expressive_high only
```

because the targeted `expressive_high-only` test set preserved reference gender in all tested samples.

However:

> **This candidate has NOT yet been implemented in the production mapping.**

No mapping change should be recorded as completed until team alignment and a real-pipeline targeted validation are performed.

---

# 20. Important limitation of the current experiment

The experiment primarily used **human-perceived gender/identity consistency** as the fast diagnostic.

Passing:

```text
male reference → male output
female reference → female output
```

does **not** prove that the output is the same speaker identity.

It only rules out the most obvious catastrophic identity collapse.

A future automated speaker-similarity/QC system would need a more precise speaker verification measure.

Therefore the current result should be described as:

> `expressive_high-only` passed the current targeted perceptual gender/identity smoke test.

Not:

> `expressive_high-only` is mathematically guaranteed to preserve speaker identity.

---

# 21. Current Controlled TTS mapping constraint

Current mapping supports `combination_overrides`.

Existing example:

```text
low + warm
→ <|emotion:contentment|>
```

However the current adapter treats an override as a **joint realization** and gives both required controls the same override token/status.

This matters because a possible future safety fallback:

```text
high + angry
→ expressive_high only
```

must NOT falsely claim:

```text
angry = realized
```

if the anger token is intentionally suppressed.

If this candidate is implemented, metadata should truthfully distinguish:

```text
arousal.high
→ context_dependent_proxy
→ <|prosody:expressive_high|>

coarse_affect.angry
→ not_realized / explicitly degraded
→ no anger token
```

This likely requires a small extension to support **per-control realization metadata** for a combination override.

No such production change has been made yet.

---

# 22. Runtime parameters to keep frozen

Unless new evidence appears, do not reopen these during the current audio-quality investigation:

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

Reason:

- memory stability improved;
- fresh pilot completed 30/30;
- no OOM/retry/500 in the final run;
- speaker-identity failure reproduced independently of the 1024/runtime tuning;
- further runtime tuning would mix unrelated concerns.

---

# 23. Current production gate

Current engineering gate:

```text
Infrastructure/runtime:
PASS

Transport/render completion:
PASS

OOM reliability:
PASS

Performance:
PASS / acceptable
(+3.16% wall-clock vs fresh 2048 pilot; no confirmed token/s regression)

Termination robustness:
KNOWN FAILURE MODE
contained by 1024 safety ceiling but not semantically solved

Audio quality:
FAIL / NOT FULLY GATED

Speaker-control composition:
FAIL for high+angry joint native controls

Main corpus:
HOLD
```

---

# 24. Proposed discussion with the model/acoustic-control teammate

This section is intended as the basis for the next team discussion.

## 24.1 Start with the shared problem statement

Suggested framing:

> We now have a production GPU/runtime that is stable enough to freeze. The remaining blocker is not infrastructure; it is acoustic-control correctness. In a real pilot, one `high + angry` turn assigned to a male reference consistently renders as female. We reproduced this outside the full pipeline, so we should decide how the control mapping should degrade when control fidelity conflicts with speaker identity.

This keeps the discussion focused on evidence rather than ownership.

---

## 24.2 Evidence to present

### Reference identity

```text
spk_012
→ VCTK p252
→ male
```

### Full-pipeline observation

```text
corpus_v1_000621 / source_turn_id=6 / turn_007.wav
expected: male spk_012
observed: female
```

### A/B result

```text
no tag                          3/3 M
expressive_high only            3/3 M
anger only                      3/3 M
expressive_high + anger         3/3 F
same combo, different text      3/3 F
same combo, another male spk    2/3 F
```

### Reverse-order result

```text
anger + expressive_high
same bad speaker/text           2/3 F
other text                      3/3 M
other male speaker              3/3 M
```

### Anger-only cross-speaker result

```text
spk012 / original text          3/3 M
spk012 / other text             3/3 M
other male / original text      2/3 F
female reference                3/3 F
```

### Expressive-high-only cross-speaker result

```text
male reference                  3/3 M
female reference                3/3 F
plus earlier spk012             3/3 M
```

Current targeted expressive-high-only total:

```text
9/9 preserved expected reference gender
```

---

# 25. Main point to argue

The key engineering principle should be:

> **Speaker identity is a hard corpus invariant; fine-grained acoustic control is allowed to degrade when the model cannot safely satisfy both.**

Reason:

A corpus turn that is slightly less angry/high than requested is still usable and auditable.

A corpus turn where the assigned speaker changes identity is not a valid realization of the scenario speaker.

This gives the priority:

```text
1. speaker identity / intelligibility
2. stable synthesis
3. coarse acoustic-control fidelity
4. optional finer expressiveness
```

---

# 26. Position on the current `high + angry` mapping

Do not argue that:

```text
anger is globally broken
```

because the experiments do not support that.

Do not argue that:

```text
expressive_high is guaranteed safe
```

because the test set is still limited.

The evidence-supported claim is narrower:

> The joint `expressive_high + anger` composition is unsafe in the tested Higgs reference-conditioned production setting.

And:

> `expressive_high-only` is currently the strongest tested candidate fallback because it preserved reference gender across the targeted male/female tests, while `anger-only` failed a cross-speaker test.

---

# 27. Questions to resolve with teammate

The discussion should reach explicit agreement on these points:

### Q1. What is the production priority?

Proposed answer:

```text
speaker identity > exact realization of every requested acoustic field
```

---

### Q2. Is degrading one required acoustic field acceptable?

For:

```text
arousal=high
affect=angry
```

candidate:

```text
realize high arousal proxy
suppress anger token
record angry as not_realized/degraded
```

Need team agreement because this changes the semantics of the Controlled TTS contract.

---

### Q3. Should the current mapping version be amended or versioned?

Because `controlled_tts_v1` was previously frozen as provisional production mapping.

Possible approaches:

```text
A. small v1 safety override with documented production evidence
B. create a mapping revision/version
```

Prefer the smallest approach consistent with the team's traceability expectations.

---

### Q4. What evidence threshold is enough?

Current evidence is targeted perceptual A/B testing.

Possible agreement:

```text
- accept current targeted tests for the immediate safety fallback;
- record them as new evidence;
- require small real-pipeline smoke validation;
- do not claim universal model behaviour.
```

---

### Q5. Do we need automated speaker verification now?

Recommended current position:

```text
Not required before deciding the mapping fallback,
but speaker consistency should become a production QC concern.
```

Do not block the mapping decision on building a large new ML subsystem.

---

# 28. What not to mix into this discussion

Keep these separate:

## Runtime tuning

Already frozen:

```text
71680 / 1 / 1 / 1024
```

Do not reopen unless new runtime evidence requires it.

---

## Cap-hit / long silence

Separate issue:

```text
corpus_v1_000621 / turn_012
```

Needs its own QC/rejection policy.

Do not use the speaker-control mapping patch to hide or conflate the runaway-generation problem.

---

## Full speaker-verification model

Potential future QC enhancement.

Not necessary to decide the immediate `high + angry` safety policy.

---

# 29. Candidate next implementation after team agreement

**Not yet implemented.**

If the team agrees to use `expressive_high-only` as the safety fallback:

### Mapping

```text
requested:
arousal = high
affect = angry

output prefix:
<|prosody:expressive_high|>
```

### Metadata

Conceptually:

```text
arousal:
  requested: high
  status: context_dependent_proxy
  token: <|prosody:expressive_high|>

coarse_affect:
  requested: angry
  status: not_realized
  token: null

combination:
  reason: speaker_identity_safety_fallback
```

The exact schema should be implemented in a backwards-compatible way.

---

# 30. Required targeted validation after any mapping change

Do **not** immediately rerun full pilot30.

First:

```text
1. unit/offline mapping tests
2. rerender corpus_v1_000621
3. listen specifically to turn_007
4. confirm expected male spk_012 identity
5. inspect generated metadata
6. run several additional high+angry real-pipeline smoke turns
```

Expected metadata:

```text
prefix_tokens = ["<|prosody:expressive_high|>"]

<|emotion:anger|> absent

arousal = represented by high proxy
angry = explicitly not realized / degraded
```

Only after that should the main corpus gate be reconsidered.

---

# 31. Separate future QC task: cap-hit / invalid audio

The known abnormal `turn_012` remains unresolved by mapping work.

Future narrow QC should consider:

```text
generation cap-like output
trailing silence
speaker consistency / identity anomalies
```

At minimum:

```text
abnormal HTTP-200 WAV
```

must no longer silently equal:

```text
production-quality success
```

Implementation should preserve failed evidence rather than silently overwriting it.

Automatic stochastic rerender/retry should not be added casually because it affects reproducibility.

---

# 32. Repository / implementation milestones referenced during this work

Known milestones include:

```text
271e226
```

Baseline before the final runtime hardening work.

```text
01cf19cc...
```

OOM-specific one-retry implementation.

Later production state examined around:

```text
68d8faf97a993b275e84f07037b1974cdac03642
```

This document should be updated with newer exact commit SHAs whenever the current production branch moves forward.

---

# 33. Tests observed during production hardening

After the runtime/bootstrap changes:

Focused/offline set:

```text
198 passed
1 deselected
```

Full offline suite at that point:

```text
496 passed
2 failed
1 warning
```

The two failures were known pre-existing/baseline issues:

1. controlled-TTS reference parity;
2. stale historical experiment mapping expecting `max_new_tokens=2048`.

The production GPU validation was therefore still necessary despite strong offline test coverage.

---

# 34. Scope intentionally excluded from the current runtime work

The following were deliberately not introduced:

```text
concurrency > 1
worker pool
partial-dialogue resume
mem_fraction_static tuning
prefill graph changes
codec graph expansion
model replacement
quantisation changes
sampling-profile change
schema-wide refactor
unrelated parity clean-up
```

This was intentional scope control.

---

# 35. Current recommended project statement

A concise project-status statement:

> The Higgs L4 runtime has reached a stable sequential production profile at `71680 / 1 / 1 / 1024`, completing a fresh 30-dialogue pilot without OOM, retry, HTTP 500, or server failure. However, perceptual validation exposed two model-output correctness issues that are invisible to HTTP/WAV structural validation: one reproducible generation-cap/long-silence failure and one speaker-identity failure associated with composed `expressive_high + anger` controls. Targeted A/B tests indicate that the joint control composition is unsafe, while `expressive_high-only` is currently the strongest tested safety fallback. No mapping change has yet been committed; the next step is team agreement on degradation semantics, followed by a small targeted pipeline validation and a separate post-render QC guard.

---

# 36. Decision log

| Item | Current decision |
|---|---|
| L4 production environment | keep |
| Higgs model | keep |
| SGLang serving path | keep |
| `max_total_tokens=71680` | freeze |
| `max_running_requests=1` | freeze |
| `cuda_graph_max_bs=1` | freeze |
| `max_new_tokens=1024` | freeze as safety cap |
| OOM retry | keep narrow one-retry policy |
| `high + angry` current joint native tags | unsafe / do not treat as validated |
| reverse tag order | rejected as reliable production fix |
| anger-only fallback | rejected as globally safe fallback |
| expressive-high-only fallback | strongest current candidate; not implemented |
| main corpus rendering | hold |
| cap-hit QC | required follow-up |
| speaker consistency QC | required production concern |
| automatic broad rerender retry | not approved |
| full runtime retuning | not needed now |

---

# 37. Immediate next meeting outcome desired

The team should leave the discussion with one explicit answer:

> **When Higgs cannot reliably preserve speaker identity while jointly satisfying `arousal=high` and `affect=angry`, are we willing to degrade the angry control and preserve speaker identity using the currently better-performing high-arousal proxy?**

If **yes**:

```text
implement the narrow mapping safety fallback
→ targeted tests
→ rerender affected dialogue
→ small real-pipeline smoke set
```

If **no**:

A different control strategy/model policy must be proposed and validated before main-corpus rendering resumes.

---

## Maintenance note

Update this document whenever any of the following change:

- runtime profile;
- Higgs/SGLang versions;
- retry semantics;
- controlled-TTS mapping version;
- speaker pool;
- QC policy;
- production acceptance gate;
- evidence/experiment outcome;
- production commit SHA.

Do not rewrite historical results after a configuration change. Add a new dated decision/result entry so the production evolution remains auditable.
