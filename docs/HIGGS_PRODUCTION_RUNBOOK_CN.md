# Higgs 生产验证执行手册

> 正式验收项以 [`HIGGS_CLOUD_VALIDATION.md`](HIGGS_CLOUD_VALIDATION.md) 为准。本手册只解释怎样执行、记录和排错，不替代 evidence checklist。

当前 `main` 基线 `d048fae2a593c72db3fe18927845478ed789323a` 尚未完成真实 Higgs reference-conditioned Cloud runtime validation。Production integration 与 offline validation 已完成，但 Cloud gate 通过前不能宣称 Higgs production-ready。

## 1. Cloud machine 要求

准备一台能运行所选 Higgs checkpoint 与 SGLang-Omni stack 的 Linux GPU machine：

- NVIDIA GPU、匹配的 driver/CUDA userspace，显存足以加载实际 checkpoint；
- 本地磁盘可容纳 repository、model、reference、outputs 与 evidence；
- 能执行 `sgl-omni`、Python、`uv`、`ffmpeg`、`git`、`nvidia-smi`；
- loopback `127.0.0.1` 上有一个空闲端口，默认 18080；
- output/evidence directory 可写；
- 若下载依赖/model，使用项目批准的凭据与网络路径，不把 token 写入 config 或 log。

历史实验在 NVIDIA L4 上完成过未加 reference 的技术 smoke，但这不是最低硬件保证，也不是当前 production evidence。

## 2. Checkout exact main SHA

Evidence 必须对应合并后的 exact `main`，不要使用未合并 feature branch：

```bash
git switch main
git pull --ff-only
git rev-parse HEAD
git status --short
git log --oneline -5
```

把 full SHA 写入 evidence notes。运行前 worktree 应清楚说明是否 clean；若不是，保存 diff 并停止 production approval。当前文档阶段的已知基线是 `d048fae2a593c72db3fe18927845478ed789323a`，但真实验证应以届时已合并 `main` 的 SHA 为准。

## 3. Python / project environment

```bash
uv sync
uv run --with pytest pytest -q
uv run ruff check .
uv run python --version
uv --version
ffmpeg -version | head -1
```

先保存 offline test 结果。若只出现本地 optional Controlled-TTS reference checkout 的 parity formatting/content mismatch，不要修改 production contract 来隐藏；记录 reference checkout 状态并按项目决定是否移除/同步该 optional checkout。其他失败必须在 Cloud run 前解释。

## 4. Higgs / SGLang runtime

安装或 provision 已批准的 SGLang-Omni runtime。Renderer 不负责下载它。确认 server executable 是普通 executable file：

```bash
command -v sgl-omni
/absolute/path/to/sgl-omni --help
sha256sum /absolute/path/to/sgl-omni
```

Worker 会用该 executable 启动：

```text
sgl-omni serve --model-path <model_dir> --host 127.0.0.1 --port <port>
```

不要先手工常驻另一个 server 占用相同端口；production worker 必须拥有并回收它自己的 process group。

## 5. Model path

把 checkpoint 放在 Cloud local disk，并记录：model ID、absolute path、revision（若已知）、关键 artifact hash、总大小。至少确认目录存在且 worker user 可读：

```bash
test -d /absolute/path/to/higgs-model
du -sh /absolute/path/to/higgs-model
find /absolute/path/to/higgs-model -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
sha256sum /absolute/path/to/higgs-model/model.safetensors
```

若 checkpoint 分片，保存所有实际 shard 名称与 hash，或保存批准的 manifest。不要把 model 文件提交到 repository。

## 6. `config_higgs_cloud.yaml` 设置

Repository 提供 `config/config.higgs.example.yaml`；复制为本地运行配置并填绝对路径：

```bash
cp config/config.higgs.example.yaml config/config_higgs_cloud.yaml
```

关键内容：

```yaml
tts:
  engine: higgs
  higgs:
    server_executable: /absolute/path/to/sgl-omni
    model_dir: /absolute/path/to/higgs-model
    host: 127.0.0.1
    port: 18080
    startup_timeout_seconds: 900
    inference_timeout_seconds: 300
    ffmpeg_bin: ffmpeg

fade_ms: 5
telephone:
  sample_rate: 8000
  channels: 1
  high_pass_hz: 300
  low_pass_hz: 3400
  volume_db_reduction: 3
```

不要把 speaker reference 放进 config；它只来自 sidecar。保留 `engine: higgs`，不要配置 fallback。记录 config SHA：

```bash
sha256sum config/config_higgs_cloud.yaml
```

## 7. Reference candidate 准备

对每个测试 `spk_*`：

1. 选择 WAV candidate，检查可读取、内容与 transcript/identity。
2. 记录 declared path、absolute path、SHA-256、时长、sample rate、channels。
3. 在 review-only registry copy 中加入 `higgs_reference`；不要先修改 production registry。
4. 明确 candidate 尚未 approved。

```bash
sha256sum /absolute/path/to/candidate.wav
ffprobe -v error -show_entries stream=sample_rate,channels,duration \
  -of default=noprint_wrappers=1 /absolute/path/to/candidate.wav
```

CosyVoice `primary_reference` 可以被评估为 candidate，但它不自动有效，也不自动 approved 为 Higgs reference。

## 8. Review-only Higgs sidecar materialization

使用 final validation corpus、assignment rows、review-only registry 和 active list：

```bash
uv run python scripts/materialize_speaker_assignments.py \
  --input evidence/input/higgs_validation.jsonl \
  --assignments evidence/input/higgs_validation_assignments.jsonl \
  --registry evidence/input/speaker_registry.review-only.json \
  --active-speakers data/speaker_pool/vctk_v0.1/active_speakers.json \
  --manifest evidence/input/higgs_validation_sidecar.json \
  --backend higgs
```

检查每个 dialogue 的 upstream/scenario/render identity、`higgs_reference.reference_wav` 和 SHA。Sidecar 是 renderer input；renderer 运行时不再查 registry。审核通过前，文件名和 evidence notes 都应保留 `review-only`/`candidate` 含义。

## 9. 必须记录的环境身份命令

```bash
git rev-parse HEAD
git status --short
sha256sum config/config_higgs_cloud.yaml
uv run python --version
uv pip freeze
/absolute/path/to/sgl-omni --version
ffmpeg -version
nvidia-smi
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
nvcc --version
```

另记录 SGLang-Omni、SGLang、Torch、Torch CUDA build marker：

```bash
uv run python -c 'import torch; print(torch.__version__); print(torch.version.cuda)'
```

若 Higgs runtime 使用与 project 不同的 Python，请在那个环境重复 package/version 命令。保存输出到 evidence，不要把 credentials 或完整 environment secrets 写入。

## 10. Worker startup test

不要把手工 `curl` server 当作 production startup test。准备一个合法 one-turn corpus 和 sidecar，通过完整 CLI 触发 worker：

```bash
uv run 5703tts \
  --input evidence/input/one_turn.json \
  --speaker-sidecar evidence/input/higgs_validation_sidecar.json \
  --config config/config_higgs_cloud.yaml \
  --output evidence/output/startup \
  --log-dir evidence/logs/startup \
  --verbose
```

观察：worker ready、`GET /health` 成功、cold-start 时间、没有 OOM、stderr 持续输出且无 deadlock。单纯 startup 成功不等于 reference conditioning 或质量通过。

## 11. One-turn baseline

使用 approved-for-test candidate 与完整 required controls：`rate=normal`、`arousal=2`、`affect=neutral`、`pause_before=none`、`pause_within=0`、`hesitations=0`。

检查：

- worker request 到 `/v1/audio/speech` 并返回 HTTP 200 `audio/wav`；
- request 的 reference 到达 `references: [{"audio_path": ...}]`；
- turn WAV 可读取且非空；
- model speed 仍为 1.0，normal 不运行 FFmpeg atempo；
- metadata 的 source/backend/mapping/reference identity 正确；
- QC passed。

保存 turn WAV 的 SHA、bytes、duration、sample rate、channels、frames。

## 12. Control matrix

至少覆盖正式 checklist 的五行：baseline、low+warm、high+angry、sad、anxious。每个 turn 必须包含六项 required controls，不能用缺字段的人工 partial object。

逐行保存：source JSON、Higgs prefix/model input、reference、HTTP/WAV 结果、atempo 决策、pause timing、metadata planned/realization、QC。听审时只记录观察，不从成功 HTTP 推断 emotion accuracy。

重点预期：

- slow -> Higgs speed 1.0 + FFmpeg 0.85；
- normal -> Higgs speed 1.0 + no rate transform；
- fast -> Higgs speed 1.0 + FFmpeg 1.15；
- pause_before -> assembly 0/500/900 ms；
- pause_within/hesitation -> frozen planner 的实际 model input 与 metadata 一致。

## 13. Two-speaker test

为 caller/counsellor 使用两个独立、reviewed candidate：

- sidecar render speaker ID 不同；
- 每个 turn reference 与 logical role 对齐；
- 同一个 SGLang server 跨 turn 复用；
- turn WAV、clean WAV、telephone WAV、metadata、QC 全部生成；
- 人工听审 speaker separation、voice confusion、clipping、noise、truncation。

Speaker similarity 与 separation 必须由 reviewer 记录，offline tests 不能替代。

## 14. Small batch

用多个 dialogue 跑一次不带 `--resume` 的小 batch：

```bash
uv run 5703tts \
  --input evidence/input/small_batch.jsonl \
  --speaker-sidecar evidence/input/higgs_validation_sidecar.json \
  --config config/config_higgs_cloud.yaml \
  --output evidence/output/small_batch \
  --log-dir evidence/logs/small_batch \
  --verbose
```

记录 cold start、每 turn inference、总耗时、manifest summary、峰值显存、process/file-descriptor 变化。确认一个 worker/model 被复用，0 unexpected failures，且输出目录按 dialogue 分开。

## 15. Resume test

先原样重复并加 `--resume`：

```bash
uv run 5703tts \
  --input evidence/input/small_batch.jsonl \
  --speaker-sidecar evidence/input/higgs_validation_sidecar.json \
  --config config/config_higgs_cloud.yaml \
  --output evidence/output/small_batch \
  --log-dir evidence/logs/resume \
  --resume --verbose
```

预期全部 resumable dialogue 不启动 worker。然后分别控制性改变/损坏一个因素：turn WAV bytes、metadata、selected reference bytes（同时保留旧 SHA）、render-affecting config。每次只改变一个变量，确认目标 dialogue rerender，unmanaged file 保留，stale managed turn 被清理。测试后恢复 candidate/reference，不把损坏文件当 evidence sample。

## 16. Shutdown verification

正常 batch 结束后，确认 parent、worker、SGLang process group 均退出，端口释放，GPU allocation 回收：

```bash
ps -ef | grep -E 'higgs_worker|sgl-omni' | grep -v grep
ss -ltnp | grep ':18080'
nvidia-smi
```

再测试一次受控 SIGTERM/CLI interruption，确认 worker 的 SIGTERM/SIGKILL escalation 最终 reap server。不要在共享机器上用 broad `pkill` 作为验收方法。

## 17. GPU / resource inspection

在 startup 前、model ready 后、batch 中间、结束后各记录：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
ps -o pid,ppid,pgid,stat,etime,rss,cmd -C sgl-omni
lsof -p <worker-pid> | wc -l
```

关注 OOM、显存持续增长、orphan process、zombie、FD 增长和异常 server restart。发现资源增长不能仅以最终成功 manifest 忽略。

## 18. 预期 output tree

```text
evidence/output/small_batch/
  batch_result.json
  dialogue_001/
    turn_001.wav
    turn_002.wav
    dialogue_001_clean.wav
    dialogue_001_telephone.wav
    dialogue_001_metadata.json
  dialogue_002/
    ...
```

成功结束后不应残留 `*.part`、`*.higgs_raw.wav` 或 `*.higgs_processed.part.wav`。Unmanaged evidence notes 可以存在于 dialogue directory，但 rerender cleanup 不应删除它们。

## 19. Metadata / QC 检查

逐个抽查：

- `provenance.record_sha256` 与 source canonical identity；
- assignment/registry/active-speaker SHA；
- backend/mapping/implementation identity；
- generation profile 中 speed 1.0；
- actual selected references；
- source identity 与 requested controls；
- planned model input、rate plan、pause/hesitation realization；
- execution filename/status；
- speech-only timing；
- QC checks 全 true。

QC passed 只表示 structural/control integrity，不表示 intelligibility、naturalness、emotion accuracy、speaker similarity 或 clinical validity。

## 20. 常见失败类别

| 类别 | 首要检查 |
| --- | --- |
| runtime path missing | config absolute path、executable bit、model directory |
| port ownership | 18080 是否已被其他进程占用 |
| startup timeout/OOM | worker stderr、`nvidia-smi`、stack compatibility |
| health never ready | `/health` response、server exit code、model load log |
| request HTTP error | `/v1/audio/speech` response、actual SGLang API compatibility |
| reference failure | declared/resolved path、live SHA、WAV readability |
| malformed protocol | stdout 被第三方污染、worker/server lifecycle |
| stderr deadlock | parent drain thread 是否持续工作 |
| FFmpeg rate failure | executable、stderr、intermediate WAV |
| metadata/QC mismatch | prepared plan、turn order、output filenames/timing |
| resume miss | fingerprint、metadata/WAV integrity、live reference SHA |
| shutdown leak | worker/server PGID、port、GPU process |

## 21. 要保留的 evidence

- exact merged main SHA 与 clean-status record；
- config file（去 secret）及 SHA、backend identity；
- model ID/path/revision/available hashes；
- Python/package/SGLang/Torch/CUDA/driver/GPU identity；
- candidate/approved reference path、SHA、reviewer/date/status；
- startup、per-turn、batch timing；
- request/result summary、WAV technical metadata/hash；
- representative audio、metadata、QC、manifest；
- resume/corruption/cleanup/shutdown/resource evidence；
- 所有 failure/warning 的 disposition；
- perceptual review sheet 与最终 approval decision。

## 22. 不要提交的内容

- model checkpoints、download caches、large generated corpus；
- Cloud credentials、API tokens、private keys、home-directory secrets；
- 未脱敏的 full environment dump；
- routine full server logs（保留必要 excerpt/evidence summary）；
- `config/config_higgs_cloud.yaml` 中机器专用绝对路径或 secrets；
- 未获批准的 reference candidate 作为 production registry entry；
- 大规模 final source corpus（当前 repository 也未包含最终完整语料）。

## 23. Production approval 边界

只有正式 checklist 的 startup、reference-conditioned one-turn、control matrix、two-speaker、small batch/resume、shutdown/resource 和人工感知 review 全部通过，且 evidence 与 exact merged main SHA 绑定后，才能批准 Higgs production use。

HTTP 200、QC passed 或 offline tests 任何一个单独成立都不够。若 runtime protocol、reference conditioning、speaker separation、intelligibility/naturalness、resource cleanup 或 metadata integrity 未通过，状态仍是 pending。批准 reference candidate 也必须逐个落到 production registry 并重新 materialize sidecar；不能把 review-only sidecar 直接当最终交付。
