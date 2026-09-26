# Running

This is the single operational guide for local/offline checks and Google Cloud
GPU rendering. Project status and formal evidence gates are in
[VALIDATION.md](VALIDATION.md).

## Requirements

Use a normal sudo-capable user on Ubuntu 22.04 or 24.04. The reviewed Google
Cloud GPU bootstrap requires Secure Boot to be disabled. If it is enabled, the
bootstrap stops before NVIDIA installation; it does not modify firmware or
security settings. `--check` reports the condition as non-PASS/`WRONG_VERSION`.

The repository `.venv` is Python 3.11. The backend environments are deliberately
separate:

```text
.venv/                              Python 3.11, CLI and tests
third_party/CosyVoice/.venv         Python 3.10, CosyVoice worker/model
third_party/sglang-omni/.venv       Python 3.12, Higgs/SGLang worker/server
```

## Fresh Cloud bootstrap

From a fresh checkout, run:

```bash
bash scripts/setup_cloud_environment.sh
bash scripts/setup_cloud_environment.sh --check
```

The modes are:

| Mode | Scope |
| --- | --- |
| `--all` | Base, CosyVoice3, and Higgs3 (default) |
| `--base` | Ubuntu/GPU/uv/project prerequisites |
| `--cosyvoice` | Base plus CosyVoice source/environment/model |
| `--higgs` | Base plus CUDA/UCX/SGLang/Higgs model/config |
| `--check` | Read-only status report |

The script checks or installs project Python 3.11, CosyVoice Python 3.10,
Higgs/SGLang Python 3.12, CUDA/UCX, pinned backend source and model revisions,
GPU availability, and the local Higgs configuration. If a reboot is required,
reconnect and rerun the same command (or use `--no-reboot`, which exits `75`).
Record the actual repository SHA, pins, GPU, and model identity printed by the
run for evidence.

## Offline project checks

```bash
uv sync --dev
uv run 5703tts --help
uv run pytest -q
uv run ruff check .
git diff --check
```

These checks do not download models or establish GPU, reference, acoustic, or
perceptual validation.

## Prepare production chunks and a speaker sidecar

Assignments do not rewrite source JSON. Production uses the frozen all-1000
assignment at `data/speaker_assignment/v0.2/` and the 15 approved Higgs
references at `data/speaker_pool/vctk_v0.2/`. Prepare 10 chunks of 100
dialogues in canonical corpus row order:

```bash
uv run python scripts/build_production_chunks.py \
  --corpus data/final/corpus_v1_1000.jsonl \
  --assignments data/speaker_assignment/v0.2/speaker_assignments.jsonl \
  --output data/production/v0.2 \
  --chunk-size 100
```

The builder fails if the output directory exists. It selects exact frozen
assignment rows for each input chunk; it never recomputes speaker identities.
The output contains `production_manifest.json` and `chunk_000/` through
`chunk_009/`, each with `input.jsonl`, `assignments.jsonl`, and
`chunk_manifest.json`. After preparation, materialize one chunk's sidecar:

```bash
uv run python scripts/materialize_speaker_assignments.py \
  --input data/production/v0.2/chunk_000/input.jsonl \
  --assignments data/production/v0.2/chunk_000/assignments.jsonl \
  --registry data/speaker_pool/vctk_v0.2/speaker_registry.json \
  --active-speakers data/speaker_pool/vctk_v0.2/active_speakers.json \
  --manifest data/production/v0.2/chunk_000/speaker_sidecar.json \
  --backend higgs
```

`--backend higgs` requires `production_approved` Higgs references with matching
live hashes (gate commit `acdb1acfdf6ada23ed036bd9306aaf1ed3f68824`).
`--backend both` emits both contracts. CosyVoice `primary_reference` is not
Higgs approval. Pool v0.2 was frozen at
`4539983c8b8e7ecfe4eba4ff686c31083edeaba1`; assignment v0.2 at
`43b5cc880e53e455e526dd82e05966d41dcbde9f`.
Tracked pilot/review sidecars contain their original machine's `project_root`;
generate a new sidecar for each prepared chunk on the machine that renders it.

## Select one backend

`tts.engine` is `higgs` or `cosyvoice`. One run uses one backend and there is no
automatic fallback.

`config/config.yaml` is the repository template and its Higgs paths are
placeholders. For CosyVoice, use `config/config_cosyvoice.yaml`.

### Higgs3

After the Higgs bootstrap, use only the generated and checked configuration:

```text
config/config_higgs_cloud.yaml
```

The reviewed bootstrap contract is:

```text
SGLang executable: third_party/sglang-omni/.venv/bin/sgl-omni
model directory:   models/higgs-tts-3-4b
host/port:         127.0.0.1:18080
server flags:      --max-total-tokens 71680 --max-running-requests 1 --cuda-graph-max-bs 1
```

Verify the executable with:

```bash
third_party/sglang-omni/.venv/bin/sgl-omni --help
```

Do not copy `config/config.higgs.example.yaml` over the generated cloud config.
The example is a template/reference for non-bootstrap custom environments.
Custom absolute-path configurations are outside the reviewed bootstrap
contract, and `setup_cloud_environment.sh --check` will not treat them as the
standard local Higgs configuration.

Run a one-dialogue smoke with a prepared one-dialogue input and matching
approved sidecar (the paths below are operator-prepared examples):

```bash
uv run 5703tts \
  --input data/production/v0.2/smoke/input.jsonl \
  --speaker-sidecar data/production/v0.2/smoke/speaker_sidecar.json \
  --config config/config_higgs_cloud.yaml \
  --output data/output/higgs-smoke \
  --log-dir logs/higgs-smoke \
  --verbose
```

Higgs model speed is fixed at `1.0`; semantic rate is FFmpeg `atempo=0.85`,
no transform, or `1.15` for slow, normal, or fast. Runtime success is not
reference approval or perceptual validation.

### CosyVoice3

```bash
uv run 5703tts \
  --input data/final/one_dialogue.json \
  --speaker-sidecar data/speaker_sidecar.json \
  --config config/config_cosyvoice.yaml \
  --output data/output/cosyvoice-smoke \
  --log-dir logs/cosyvoice-smoke \
  --verbose
```

CosyVoice3 uses its isolated worker and `text_frontend=False`; rate is
`0.8/1.0/1.2`, affect/arousal instructions are provisional, and positive
`pause_within` fails closed during whole-dialogue preflight.

## Batch and resume

```bash
uv run 5703tts \
  --input data/final/small_batch.jsonl \
  --speaker-sidecar data/speaker_sidecar.json \
  --config config/config_higgs_cloud.yaml \
  --output data/output/small \
  --log-dir logs/small

uv run 5703tts \
  --input data/final/small_batch.jsonl \
  --speaker-sidecar data/speaker_sidecar.json \
  --config config/config_higgs_cloud.yaml \
  --output data/output/small \
  --log-dir logs/resume \
  --resume
```

For a CosyVoice batch, use `config/config_cosyvoice.yaml` in both commands.
Legacy and CosyVoice resume require a successful prior manifest entry with the
same semantic fingerprint, readable metadata/WAV artifacts, and matching live
selected reference bytes. Assessed Higgs audio uses its dialogue quality
sidecar and verifies turn hashes before preservation. A changed source,
selected backend/reference, shared audio setting, or relevant identity causes
rerender. Manifest v2 is atomically written at batch completion, not after
every dialogue.

## Outputs

Each dialogue directory contains speech-only `turn_NNN.wav`, a clean assembled
WAV, a telephone-labelled WAV, metadata, and structural QC. Telephone output is
only mono/resampling/filtering/level processing, not PSTN or codec simulation.

For offline Higgs turn-audio diagnostics:

```bash
uv run python scripts/analyze_higgs_audio_qc.py \
  --output-root data/output/higgs_pilot30_final_1024 \
  --json-out /tmp/higgs_pilot30_audio_qc.json
```

The analyzer remains an offline diagnostic report. Production Higgs rendering
also applies `higgs_audio_quality_v1`: a termination-sanity gate requiring a
final 5 s energy drop of at least 10 dB **and** a trailing body-minus-10-dB
run of at least 3.0 s. Exact decoded digital zero rejects, and unavailable
metrics reject as indeterminate. This is not general perceptual QC.

Each assessed Higgs dialogue has a `<dialogue_id>_quality.json` sidecar with
the policy, disposition, and SHA-256 of every turn WAV. It governs those WAV
bytes; `batch_result.json` governs the latest batch execution, and immutable
attempt manifests are audit history. A plain run preserves a same-identity
quality rejection before cleanup. Use `--resume` to preserve it or reassess
existing audio offline after a quality-policy change. Artifact-integrity
failure requires operator action; the pipeline does not trim, retry, or repair
audio automatically. The quality sidecar is not a synthesis checkpoint.
Render fingerprints do not cover every external SGLang launch flag, so
operators must not rely on `--resume` after unrepresented render-affecting
server changes.

Each completed batch also publishes `quality_rejected.jsonl`, an operational
queue for genuine quality rejects. It is not audit authority; batch manifests
and Higgs quality sidecars remain the execution and assessed-WAV authorities.
`--resume` reuses or reassesses existing audio; it never retries a rejected
render. A retry is an explicit fresh render into a new output namespace:

```bash
uv run python scripts/build_quality_retry_batch.py \
  --rejects data/output/production_v0.2/chunk_000/quality_rejected.jsonl \
  --corpus data/final/corpus_v1_1000.jsonl \
  --assignments data/speaker_assignment/v0.2/speaker_assignments.jsonl \
  --output data/production/v0.2_retry/chunk_000_retry01 \
  --attempt 1
```

Materialize that retry pack's sidecar with the v0.2 registry, then render its
`input.jsonl` to `data/output/production_v0.2_retry/chunk_000_retry01/`
without `--resume`. Preserve both attempts and their sidecars. The retry
builder (`e0770bf0e93f41332f6f7817ba7a5ac7c1a99ec1`) and queue
(`11d369835c729e024a3bd84d4103a77078091506`) do no automatic broad
rerendering.

After first pass and controlled retries, build a derived final acceptance and
delivery index. It reads evidence only; it does not render or retry audio:

```bash
uv run python scripts/build_production_acceptance_manifest.py \
  --corpus data/final/corpus_v1_1000.jsonl \
  --first-pass chunk_000=data/output/production_v0.2/chunk_000/batch_result.json \
  --retry chunk_000_retry01=data/output/production_v0.2_retry/chunk_000_retry01/batch_result.json \
  --output data/production/final_acceptance.json
```

Repeat `--first-pass` and `--retry` for every supplied batch; retry order is
the supplied order. Add `--require-complete` for the final all-corpus gate.
This derived index does not replace batch results or quality sidecars. The
current 30-dialogue pilot corresponds to 28 `accepted_first_pass`, 2
`accepted_after_retry`, and 30 `accepted_total`; the full corpus is unfinished.

## Troubleshooting

| Symptom | First checks |
| --- | --- |
| Secure Boot gate fails | Disable Secure Boot in the VM/firmware setup, then rerun; the script will not change it |
| Setup exits `75` | Reboot, reconnect, and rerun the same setup mode |
| CUDA/UCX/GPU gate fails | Run `--check`, inspect `nvidia-smi`, driver/CUDA/UCX identity, and GPU visibility |
| Higgs config mismatch | Use the generated `config/config_higgs_cloud.yaml`; do not overwrite it with the example |
| Higgs worker startup/port failure | Check executable path, model directory, port `18080`, worker stderr, and `nvidia-smi` |
| Wrong reference | Check sidecar field, resolved path, and live SHA-256 |
| CosyVoice preflight failure | Check positive `pause_within`, prompt WAV/text, and selected config |
| Corrupt output or unexpected rerender | Check metadata/WAV integrity, fingerprint, and selected reference SHA |
| Stale artifacts remain | Only managed direct-child names are cleaned; unmanaged files are retained |

## Runtime identity for evidence

Formal evidence must record the exact repository commit SHA, config SHA, selected
backend and mapping identity, model revision/checkpoint SHA, Python/package/
SGLang/UCX/CUDA/GPU identity, reference paths and hashes, startup and render
observations, output WAV metadata, manifest, QC, resume, and shutdown/resource
behavior. Do not infer control fidelity, perceptual quality, or clinical
usefulness from runtime completion or structural QC.
