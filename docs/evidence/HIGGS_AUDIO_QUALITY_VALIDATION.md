# Higgs audio-quality production validation

**Final status:** IMPLEMENTED + TARGETED REAL-AUDIO VALIDATED

This is a historical validation snapshot. Its conclusion that a 30-dialogue
rerun was unnecessary applied at that point; a later production pilot did run.
See [current validation](../VALIDATION.md) and the
[production handoff](../CloudProductionLog.md) for current overall status.

The production validation commit was `ac4745e` (`fix Higgs worker executable permission`). The quality implementation was already present in the preceding production code state. This commit changed only `src/tts5703/backends/higgs_worker.py` from Git mode `100644` to `100755`. After pulling it on Google Cloud, `git ls-files` reported `100755`, the file was executable, `uv run ruff check .` reported `All checks passed`, and `uv run ruff format --check .` reported `74 files already formatted`. Cloud full pytest was not run; pytest is unavailable in the production environment.

## Frozen production runtime

Google Cloud NVIDIA L4 (~22.49 GiB VRAM), model `bosonai/higgs-tts-3-4b`:

| Setting | Value | Setting | Value |
| --- | ---: | --- | ---: |
| `max_total_tokens` | 71680 | `max_running_requests` | 1 |
| `cuda_graph_max_bs` | 1 | `max_new_tokens` | 1024 |
| `temperature` | 0.8 | `top_p` | 0.8 |
| `top_k` | 30 | `stream` | false |
| `speed` | 1.0 | | |

Validation did not change runtime, sampling, mapping, speaker pool, model, cleanup semantics, resume semantics, or generation parameters.

## Quality policy

Policy ID: `higgs_audio_quality_v1`
Current canonical policy SHA256: `134cc23da9efd8b1f7e3445232033c369675780512aa3b1707908b2f8adb7b52`

The narrow termination predicate is `tail_5_drop_db >= 10.0` **AND** trailing body-minus-10-dB duration `>= 3.0` seconds. A positive infinite tail drop from a finite body followed by an exactly-zero tail is valid and satisfies the drop threshold. Whole-file exact digital zero is rejected separately. NaN or unavailable required metrics remain indeterminate. This is a termination-sanity gate, not a general perceptual-quality model or a measured false-positive-rate claim.

## Real-audio evidence

### Historical bad artifact and terminal rejection

Dialogue `corpus_v1_000621`, known problematic `turn_012.wav`, fresh 1024 historical artifact: duration `40.68 s`; `body_dbfs=-23.23626327287063`; `tail_5_drop_db=36.54642137716846 dB`; trailing body-minus-10-dB duration `19.93 s`; ordinal `12`; `source_turn_id=11`.

The historical production artifact was copied to an isolated smoke root without a quality sidecar. A `--resume` run reassessed it offline: batch `status=failure`, `rendered=0`, `resumed=0`, `failed=1`, exit code `1`, elapsed ~`0.17 s`. The generated sidecar recorded `outcome=reject`, `assessment=reassessed`, `issue=abnormal_tail`, `action=quality_rejected`, `status=failed`. All WAV hashes before and after were byte-identical.

The rejected same-fingerprint artifact was then terminal on both rerun paths. Plain run: `status=failure`, `rendered=0`, `resumed=0`, `failed=1`, exit code `1`, elapsed ~`0.04 s`. `--resume`: the same status and counts, exit code `1`, elapsed ~`0.08 s`. The complete dialogue artifact hash set remained unchanged on each run. This validates legacy success → offline reassessment → quality rejection → preservation, and terminal rejection for plain and resume runs.

### Fresh normal real-GPU PASS and resume

Dialogue `corpus_v1_000000` has 15 turns. On the real L4, the Higgs worker started, all 15 turn requests succeeded, and the pipeline completed with `passed=True`. Batch: `status=success`, `rendered=1`, `resumed=0`, `failed=0`, exit code `0`, elapsed ~`189.49 s`.

The durable sidecar recorded `outcome=pass`, `assessment=new_render`, `issues=[]`, `action=rendered`, `status=success`, and `turn_wav_sha256` for all 15 turn WAVs. Render fingerprint: `88c8de500f7b58aafd359783696d71ce5dd5b9d4d841ca17911c3aa777ece029`. Policy SHA256: `134cc23da9efd8b1f7e3445232033c369675780512aa3b1707908b2f8adb7b52`.

Same-fingerprint `--resume`: `status=success`, `rendered=0`, `resumed=1`, `failed=0`, exit code `0`, elapsed ~`0.06 s`; the complete dialogue artifact hash set was unchanged. This validates fresh render → quality PASS → durable sidecar → resume without GPU → byte-identical artifacts.

### Artifact-integrity fail-closed check

A copy of the PASS artifact for `corpus_v1_000000` was used. One byte of `turn_001.wav` was deliberately modified: SHA256 changed from `9468d079884e5ea8cd31ccefda396305e247772576d3b0f1e17536c1d9aca059` to `c55be5c500ba4cb0dbe9415eb461e71ec201ed422f46c6c28ee8e593f51aebbe`.

The following `--resume` returned `status=failure`, `rendered=0`, `resumed=0`, `failed=1`, exit code `1`, elapsed ~`0.01 s`; `summary.quality_integrity_failed=1`, `summary.quality_rejected=0`, `action=quality_rejected`, `status=failed`. Error type was `QualityEvidenceError`, with message `Turn WAV hash changed: turn_001.wav`. Quality authority still had `outcome=pass`; `quality_observation` had `outcome=indeterminate` and `issue=artifact_integrity_failure`.

The modified WAV stayed modified; no cleanup or regeneration occurred. The original PASS quality sidecar stayed byte-identical, with SHA256 `fe3edc9be4b766a3e56fb5997805bc22dc8e3c031caad5052a10395c670d511e`. This validates hash mismatch → fail closed → no cleanup → no GPU → preservation of prior durable quality authority.

## Validation matrix

| Scenario | Expected | Observed | GPU Used | Artifact Preservation | Result |
| --- | --- | --- | --- | --- | --- |
| Historical bad / legacy migration | Reassess and reject | `abnormal_tail`; failure ~0.17 s | No | WAV hashes unchanged | PASS |
| Rejected / plain rerun | Terminal failure | 0 rendered/resumed; failure ~0.04 s | No | Complete hash set unchanged | PASS |
| Rejected / resume | Terminal failure | 0 rendered/resumed; failure ~0.08 s | No | Complete hash set unchanged | PASS |
| Fresh normal / real GPU | Render and pass | 15 turns; success ~189.49 s | Yes | Sidecar records all 15 WAV hashes | PASS |
| PASS / resume | Resume without render | 1 resumed; success ~0.06 s | No | Complete hash set unchanged | PASS |
| PASS sidecar + WAV hash mismatch | Fail closed | `QualityEvidenceError`; failure ~0.01 s | No | Modified WAV and original sidecar preserved | PASS |
| Cloud static checks | Executable worker; clean lint/format | Git mode 100755; Ruff checks passed | No | No artifact mutation | PASS |

## Checks and runtime observations

Offline/local implementation validation: focused audio-quality + batch `151 passed`; full offline pytest `550 passed`, with 2 unchanged known baseline parity failures: (1) controlled-TTS contract byte parity and (2) frozen experiment-source `max_new_tokens` parity. Ruff check, Ruff format check, and `git diff --check` passed.

During the real GPU smoke run, KV cache token capacity was `71680`, decode CUDA graph captured `bs=1`, and generation throughput was around 31 token/s after warmup. There was no OOM, HTTP 500, worker crash, or retry storm. Non-blocking warnings were: nixl unavailable; no `generation_config.json`; torchao import warning; auto-selected prefill CUDA graph disabled/fallback due to free-memory safety threshold; codec CUDA graph misses for frame counts outside the captured range; and multiprocessing `resource_tracker` semaphore warning at shutdown. The dialogue completed successfully; these warnings are not classified as regressions.

## Targeted L4 performance validation

After commit `0357905` (`add Higgs production timing instrumentation`), a fresh render of `corpus_v1_000000` (15 turns) on Google Cloud NVIDIA L4 completed with `run_exit=0`, batch `status=success`, `rendered=1`, `failed=0`, `quality=pass`, and 19 total artifacts. Dialogue `total_sec=188.627`; batch `duration_seconds=188.635`.

| Timed work | Observed seconds | Share of dialogue wall time |
| --- | ---: | ---: |
| Turn synthesis total | 186.779 | 99.02% |
| Worker cold startup | 58.928 | 31.24% |
| Sum of 15 inference `request_sec` values | 127.825 | 67.77% |

Other dialogue stages: `preflight_sec=0.010`, `assemble_sec=0.046`, `clean_export_sec=0.004`, `telephone_sec=1.664`, `metadata_sec=0.008`, `qc_sec=0.013`, `quality_sec=0.081`. Per-turn `request_sec`: minimum `4.840`, median `8.181`, mean `8.522`, maximum `13.405` seconds. For warm turns 2–15, summed `request_sec=119.520` and `total_sec=119.542`; non-request overhead was ~`0.022` seconds total, or ~`1.6 ms` per turn.

Approximately 99% of wall time was Higgs/SGLang startup and inference. No material pipeline-side bottleneck was found; assembly, WAV publication, metadata, structural QC, and quality QC were small beside inference. Worker startup is a one-time cold-start cost for a continuous batch because the worker/model is cached and reused across turns and dialogues while runtime identity is unchanged. The first turn's `total_sec` includes that nested startup duration: do not add startup time to first-turn total time. Material acceleration would require changes to the model/runtime execution strategy, such as concurrency, batching, or runtime/model changes, outside the frozen validated production scope. This single cold-dialogue run does not establish general throughput, concurrent-rendering, or full-corpus performance, or guarantee a speedup.

**Performance conclusion:** Targeted L4 timing validation found no material pipeline-side performance bottleneck. For the tested cold single-dialogue render, 99.02% of wall time was spent in Higgs synthesis. Further material acceleration would require changes to the frozen model/runtime execution strategy rather than optimization of pipeline orchestration.

## Scope and production readiness

This establishes targeted production correctness for the new Higgs audio termination QC and quality-state lifecycle. It does **not** establish general perceptual audio quality, validate concurrent rendering, validate every dialogue in the corpus under the new policy, or justify changing the frozen runtime. A full 30-dialogue rerun was not needed after targeted validation: the new logic was exercised on a known bad historical artifact, a fresh real-GPU passing artifact, resume behavior, terminal reject behavior, and deliberate artifact corruption.

**Production readiness conclusion:** The Higgs narrow audio termination QC is implemented and targeted-real-audio validated for the tested production lifecycle paths. Current state is suitable for the frozen sequential production workflow, subject to the explicitly excluded scope above.
