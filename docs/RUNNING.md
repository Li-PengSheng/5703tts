# Running

This is the operational guide. The current evidence boundary is in
[CURRENT_STATUS.md](CURRENT_STATUS.md); this page does not turn a successful
command into a production-approval claim.

## 1. Offline project environment

The project requires Python 3.11 and uses the repository `.venv`:

```bash
uv sync
uv run 5703tts --help
uv run ruff check .
uv run --with pytest pytest -q
```

The test suite is offline/static: it does not download models or require a GPU.

## 2. Fresh Google Cloud / Ubuntu setup

Run from the repository checkout as a normal sudo-capable user, not root:

```bash
bash scripts/setup_cloud_environment.sh
bash scripts/setup_cloud_environment.sh --check
```

Modes are:

| Mode | Scope |
| --- | --- |
| `--all` | Base, CosyVoice3, and Higgs3 (default) |
| `--base` | Ubuntu/GPU/uv/project prerequisites only |
| `--cosyvoice` | Verify base, then CosyVoice source/environment/model |
| `--higgs` | Verify base, then CUDA/UCX/SGLang/Higgs model/config |
| `--check` | Read-only status report; changes nothing |

The current Google Cloud GPU bootstrap targets Ubuntu 22.04 or 24.04 and
requires Secure Boot to be disabled. If Secure Boot is enabled, the bootstrap
stops before NVIDIA installation; it does not modify firmware or security
settings. `--check` reports the condition as non-PASS/`WRONG_VERSION` rather
than silently changing it. The script accepts `--no-reboot` and returns exit
`75` when a reboot is required. It checks project Python 3.11,
CosyVoice Python 3.10, Higgs Python 3.12, CUDA 13, UCX with CUDA, SGLang-Omni,
GPU availability, model revisions/SHA, and local Higgs config. Record the
actual SHA and versions printed by the run for evidence.

The three Python environments are intentionally separate:

```text
project .venv                       Python 3.11, CLI/tests
third_party/CosyVoice/.venv         Python 3.10, CosyVoice worker/model
third_party/sglang-omni/.venv       Python 3.12, Higgs/SGLang worker/server
```

## 3. Prepare input and speakers

Assign logical roles without editing source JSON:

```bash
uv run python scripts/assign_dialogue_speakers.py \
  --input data/final/dialogues.jsonl
```

Materialize a backend-specific sidecar:

```bash
uv run python scripts/materialize_speaker_assignments.py \
  --input data/final/dialogues.jsonl \
  --assignments data/speaker_pool/vctk_v0.1/speaker_assignments.jsonl \
  --registry data/speaker_pool/vctk_v0.1/speaker_registry.json \
  --active-speakers data/speaker_pool/vctk_v0.1/active_speakers.json \
  --manifest data/speaker_sidecar.json \
  --backend cosyvoice
```

Use `--backend higgs` only with a registry that has separately approved,
hash-verified `higgs_reference` entries. `--backend both` emits both contracts.
CosyVoice `primary_reference` is not Higgs approval.

## 4. Select exactly one backend

Higgs default template:

```yaml
tts:
  engine: higgs
```

CosyVoice explicit configuration:

```bash
cp config/config_cosyvoice.yaml /tmp/config_cosyvoice.yaml
```

Then pass that config (or set the same values in a local config). There is no
automatic fallback if the selected backend fails.

## 5. CosyVoice3 run

The checked-in CosyVoice config uses its own Python worker, repository, and
model directory. Run a one-dialogue smoke first:

```bash
uv run 5703tts \
  --input data/final/one_dialogue.json \
  --speaker-sidecar data/speaker_sidecar.json \
  --config config/config_cosyvoice.yaml \
  --output data/output/cosyvoice-smoke \
  --log-dir logs/cosyvoice-smoke \
  --verbose
```

Positive `pause_within` fails during whole-dialogue preflight. A successful
CosyVoice run proves execution and structural output, not affect/arousal
fidelity or speaker/perceptual quality.

## 6. Higgs3 run

After `--higgs` setup, the script writes/verifies the local ignored
`config/config_higgs_cloud.yaml`. Use an approved-reference sidecar:

```bash
uv run 5703tts \
  --input evidence/input/one_turn.json \
  --speaker-sidecar evidence/input/higgs_sidecar.json \
  --config config/config_higgs_cloud.yaml \
  --output evidence/output/higgs-smoke \
  --log-dir evidence/logs/higgs-smoke \
  --verbose
```

The worker owns the SGLang process group, waits for health, reuses one loaded
server across turns, and keeps stdout protocol-only. A real runtime success is
not equivalent to reference approval or perceptual validation.

## 7. Small batch and resume

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

Resume needs a successful prior manifest entry with the same semantic
fingerprint, readable metadata/WAV artifacts, and matching live selected
reference bytes. A changed selected backend, selected reference, shared audio
settings, source content, or relevant identity causes rerender. The manifest is
written atomically at batch completion, not after every dialogue.

## 8. Output and QC

Each dialogue directory contains speech-only `turn_NNN.wav`,
`<dialogue_id>_clean.wav`, `<dialogue_id>_telephone.wav`, and metadata. The
telephone-labelled file is only mono/resampling/high-pass/low-pass/level
processing; it is not a codec or PSTN simulation. QC is structural/control
integrity only.

## 9. Troubleshooting

| Symptom | First checks |
| --- | --- |
| setup exits `75` | Reboot, reconnect, rerun the same setup command |
| backend paths missing | `scripts/setup_cloud_environment.sh --check`, config paths, executable/model presence |
| Higgs port/startup failure | Port `18080`, worker stderr, `nvidia-smi`, SGLang compatibility |
| reference mismatch | Sidecar selected field, resolved path, live SHA-256 |
| CosyVoice preflight failure | `pause_within > 0`, prompt WAV/text, selected config |
| QC failure | Metadata/planned plan, turn order, WAV readability, timing |
| resume unexpectedly rerenders | Fingerprint, metadata/WAV integrity, selected reference SHA |
| stale artifacts remain | Only managed direct-child names are cleaned; unmanaged files are retained |

## 10. Evidence capture

Record the exact repository SHA, config SHA, backend and mapping identity, model
revision/SHA, Python/package/CUDA/GPU versions, reference IDs and hashes,
startup/per-turn/batch timing, output WAV metadata, manifest, QC, resume and
shutdown observations. Do not infer perceptual or clinical claims from HTTP
success, QC, or runtime completion alone.
