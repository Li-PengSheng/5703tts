# 安装与运行（中文操作版）

本页是 [RUNNING.md](RUNNING.md) 的中文执行版。当前状态以
[CURRENT_STATUS.md](CURRENT_STATUS.md) 为准。本文不把 runtime 成功写成
production approval，也不把 CosyVoice 当作自动 fallback。

## 1. Fresh Google Cloud / Ubuntu bootstrap

在 fresh Ubuntu 22.04/24.04 GPU VM 上，以普通 sudo-capable SSH 用户执行：

```bash
cd /path/to/5703tts
bash scripts/setup_cloud_environment.sh
```

脚本支持以下模式：

```text
--all        base + CosyVoice3 + Higgs3（默认）
--base       只安装/检查 base、GPU、uv、project env
--cosyvoice  检查 base 后安装 CosyVoice3
--higgs      检查 base 后安装 CUDA/UCX/SGLang/Higgs3
--check      只读检查，不安装、不修改
```

如果安装驱动或 CUDA 需要重启，脚本会给出相同命令；也可以使用
`--no-reboot`，它在需要重启时以 exit code `75` 退出。重连后重新执行同一
命令。

当前 Google Cloud GPU bootstrap 面向 Ubuntu 22.04/24.04，并要求 Secure Boot
为 `disabled`。如果仍为 `enabled`，bootstrap 会在 NVIDIA 安装前停止；它不
会自动修改 firmware 或其他 security setting。`--check` 会将该条件报告为非
PASS/`WRONG_VERSION`。检查结果还覆盖磁盘、NVIDIA、项目 Python 3.11、
CosyVoice Python 3.10、Higgs/SGLang Python 3.12、CUDA 13、CUDA-enabled UCX、
模型 revision/SHA、GPU 可见性、Higgs 本地 config 和端口 `18080`。

安装后建议：

```bash
bash scripts/setup_cloud_environment.sh --check
```

正式 evidence 必须记录脚本打印的实际 repository SHA 和实际 package/model/GPU
身份；旧开发 SHA 不是 fresh install 的长期要求。

## 2. 三个 Python 环境

```text
.venv/                              Python 3.11：5703tts、tests
third_party/CosyVoice/.venv/        Python 3.10：CosyVoice worker/model
third_party/sglang-omni/.venv/      Python 3.12：Higgs/SGLang worker/server
```

项目环境只负责 CLI、validation、planning、assembly、metadata 和 QC；两个
backend 通过各自 worker 使用独立环境。

## 3. 离线检查

```bash
uv sync
uv run 5703tts --help
uv run ruff check .
uv run --with pytest pytest -q
git diff --check
```

这些检查不下载模型，也不证明 GPU、speaker reference、acoustic fidelity 或
perceptual quality。

## 4. Speaker sidecar

先生成不改写 source JSON 的 deterministic assignment：

```bash
uv run python scripts/assign_dialogue_speakers.py \
  --input data/final/dialogues.jsonl
```

再按所选 backend 生成 sidecar：

```bash
uv run python scripts/materialize_speaker_assignments.py \
  --input data/final/dialogues.jsonl \
  --assignments data/speaker_pool/vctk_v0.1/speaker_assignments.jsonl \
  --registry data/speaker_pool/vctk_v0.1/speaker_registry.json \
  --active-speakers data/speaker_pool/vctk_v0.1/active_speakers.json \
  --manifest data/speaker_sidecar.json \
  --backend cosyvoice
```

`--backend` 可取 `higgs`、`cosyvoice`、`both`。当前 registry 的
`primary_reference` 是 CosyVoice contract；没有正式 approved 的
`higgs_reference` 时，不能用 `--backend higgs` 生成 production sidecar。

四层 speaker identity 始终分开：

```text
upstream: User / Listener
logical: caller / counsellor
scenario: C001 / L001
production: spk_001 / spk_008
```

## 5. 选择 backend

`config/config.yaml` 默认 `tts.engine: higgs`，并含外部 runtime placeholder。
CosyVoice 必须显式选择：

```yaml
tts:
  engine: cosyvoice
```

可以直接使用 `config/config_cosyvoice.yaml`。一次 run 只有一个 backend；
Higgs 失败不会自动转 CosyVoice。

## 6. CosyVoice3 smoke

```bash
uv run 5703tts \
  --input data/final/one_dialogue.json \
  --speaker-sidecar data/speaker_sidecar.json \
  --config config/config_cosyvoice.yaml \
  --output data/output/cosyvoice-smoke \
  --log-dir logs/cosyvoice-smoke \
  --verbose
```

CosyVoice3 使用独立 worker，`text_frontend=False`；rate 为 `0.8/1.0/1.2`，
arousal/affect 是 provisional instruction mapping。`pause_within > 0` 会在
whole-dialogue preflight fail closed，合成不会开始。

## 7. Higgs3 smoke

`--higgs` 安装模式会写入/验证本地 ignored 文件
`config/config_higgs_cloud.yaml`。使用含 approved `higgs_reference` 的 sidecar：

```bash
uv run 5703tts \
  --input evidence/input/one_turn.json \
  --speaker-sidecar evidence/input/higgs_sidecar.json \
  --config config/config_higgs_cloud.yaml \
  --output evidence/output/higgs-smoke \
  --log-dir evidence/logs/higgs-smoke \
  --verbose
```

Higgs model speed 固定为 `1.0`；`slow`/`fast` 由 FFmpeg `atempo=0.85/1.15`
实现，`normal` 不做 rate transform。`pause_before` 在 shared assembly 中是
`0/500/900 ms`，不写入 turn WAV。

## 8. Small batch 与 resume

```bash
uv run 5703tts \
  --input data/final/small_batch.jsonl \
  --speaker-sidecar data/speaker_sidecar.json \
  --config config/config.yaml \
  --output data/output/small \
  --resume
```

resume 需要旧 manifest 成功、semantic fingerprint 相同、metadata/WAV 可读且
selected reference 当前 bytes 与 sidecar SHA 一致。只要 source、selected
backend、selected reference、shared audio config 或 backend identity 改变，目标
dialogue 就会 rerender。manifest v2 在 batch 结束时原子写入，不是逐 dialogue
checkpoint。

## 9. 产物与边界

每个 dialogue 输出 speech-only `turn_NNN.wav`、clean WAV、telephone-labelled
WAV、metadata。后者只做 mono、resample、high/low-pass 和 level reduction，
不是完整 PSTN/codec simulation。QC 只证明 structural/control integrity，不证明
emotion、speaker similarity、naturalness、intelligibility 或 clinical usefulness。

## 10. Evidence 记录

记录实际执行 SHA、config SHA、backend/mapping identity、model revision/SHA、
Python/CUDA/GPU/package、reference path/SHA、startup/per-turn/batch timing、WAV
metadata、manifest、QC、resume、shutdown/resource observations。不要把 HTTP 成功、
QC passed 或 runtime completion 单独写成 production ready。
