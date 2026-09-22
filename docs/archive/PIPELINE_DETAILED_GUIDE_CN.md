# 5703tts 生产流水线详细指南（归档）

> Archived historical guide. 截至本文归档时的说明仅供历史参考；当前项目状态请查看 [../CURRENT_STATUS.md](../CURRENT_STATUS.md)。当前设计与运行请查看 [../ARCHITECTURE.md](../ARCHITECTURE.md)、[../DATA_CONTRACTS.md](../DATA_CONTRACTS.md) 和 [../RUNNING.md](../RUNNING.md)。

## 1. 项目目的

`5703tts` 把 Controlled-TTS-v1 的最终对话 JSON/JSONL 转换为可审计的双人对话音频。它不仅生成 WAV，还保存源记录、说话人、控制计划、执行结果、时间戳、后端身份和批次状态之间的关系，供交付、复现、失败定位和断点续跑使用。

## 2. 为什么需要这条流水线

上游数据描述“谁说什么、希望怎样说”；TTS 后端只接受自己的推理参数。流水线在两者之间建立稳定边界：严格拒绝歧义输入，把上游身份与生产音色分离，冻结后端执行计划，在任何 GPU 请求前检查完整对话，最后把音频、metadata、QC 和 manifest 绑定在一起。

它不负责生成上游语料，不修改最终源记录，也不把感知质量结论包装成结构化测试结论。

## 3. 截至本文归档时的后端政策

- Higgs 是 production primary。
- CosyVoice3 是显式选择的 backup/secondary。
- `tts.engine` 为整个 run 选择一个后端。
- 不存在 Higgs 失败后自动转 CosyVoice 的 fallback。
- 两种后端的 reference contract 独立，不能互换。

## 4. 完整架构图

```text
final JSON / JSONL
        |
        v
InputRecord（源快照 + canonical record SHA）
        |
        v
严格 final contract validation
        |
        v
known-issue exclusion policy
        |
        v
speaker sidecar（source identity -> production spk_* + selected reference）
        |
        v
CanonicalDialogue / CanonicalTurn
        |
        v
backend-specific planner
        |
        v
PreparedDialogue / HiggsPreparedTurn | CosyVoicePreparedTurn
        |
        v
whole-dialogue preflight（所有 turn 先通过）
        |
        v
backend synthesis（逐 turn、顺序执行）
        |
        v
TurnRenderResult + speech-only turn_NNN.wav
        |
        v
assembly（插入 pause_before，生成 timestamp）
        |
        +--> <dialogue_id>_clean.wav
        |
        +--> telephone processing --> <dialogue_id>_telephone.wav
        |
        v
<dialogue_id>_metadata.json
        |
        v
structural/control-integrity QC
        |
        v
batch_result.json（manifest v2）
        |
        v
semantic resume + live artifact/reference integrity
```

## 5. 源数据 contract

生产入口只接受 final nested contract。`.json` 是一个 dialogue；`.jsonl` 的每个非空行是一个 dialogue。CLI 对目录只发现 direct-child 的 `.json`/`.jsonl`，不递归。读取层拒绝 duplicate keys、NaN、Infinity、非对象记录与空 `dialogue_id`。JSONL 某行损坏会形成 record-level `input_error`，不会吞掉后续行。

每个 turn 必须包含 `turn_id`、`speaker`、非空 `text`、`labels` 和 `acoustic.required` 的六项控制：

- `rate`
- `arousal`
- `affect`
- `pause_before`
- `pause_within`
- `hesitations`

`best_effort` 可携带 `affect_fine`、`volume`、`flattened_affect`、`events`。保存请求不等于后端支持或已实现这些效果。legacy/v0.2 等任意旧 shape 不会被自动适配。

## 6. InputRecord 与 record SHA

`InputRecord` 是渲染身份使用的 immutable-ish 源快照：构造时 deep copy，读取 `raw` 时再次返回 copy。`record_sha256` 对 canonical JSON 内容计算，key 排序、紧凑分隔、UTF-8、禁止 NaN；它不包含文件路径、JSON 排版或 JSONL 行号。因此相同语义内容移动文件或重新格式化不会改变源身份。

## 7. role 与 speaker 身份模型

四个概念不能混用：

| 名称 | 示例 | 所有者 |
| --- | --- | --- |
| upstream role | `User` / `Listener` | source turn |
| logical role | `caller` / `counsellor` | contract normalization |
| scenario speaker ID | `C001` / `L001` | source `scenario.speakers` |
| render speaker ID | `spk_008` | production sidecar |

映射固定为 `User -> caller`、`Listener -> counsellor`。scenario ID 是上游叙事身份；`spk_*` 是生产音色身份，两者必须保持可追溯但不能相等或互相覆盖。

## 8. Speaker sidecar 设计

Sidecar 由 `scripts/materialize_speaker_assignments.py` 生成。输入是 final dialogue source、assignment rows、speaker registry、active speaker list；输出是 backend-aware dialogue sidecar。脚本不重写源 JSON。

每个 dialogue role 保存 upstream role、scenario speaker ID、render speaker ID 和目标后端 reference。Higgs 使用：

```json
"higgs_reference": {
  "reference_wav": "refs/approved_higgs.wav",
  "sha256": "<64 hex>"
}
```

CosyVoice 使用：

```json
"cosyvoice_reference": {
  "prompt_wav": "refs/cv3_prompt.wav",
  "prompt_text": "You are a helpful assistant.<|endofprompt|>Exact transcript.",
  "sha256": "<64 hex>"
}
```

声明的相对路径用于 provenance；相对 project root 解析后的路径用于运行时读取。`verify_file=True` 会在生产使用前对当前 bytes 重新算 SHA。CosyVoice `primary_reference` 即使可作为候选，也不自动成为 approved Higgs reference。

## 9. CanonicalDialogue 的理由

`CanonicalTurn` 把最终源 contract 和 sidecar 对齐后，保存 text、labels、四种身份、raw required/best-effort 请求、normalized rate/arousal/affect/pause/hesitation。它不包含 Higgs model input、native token、CosyVoice instruction、numeric speed 或 worker request。

这个中间层让两个后端共享同一套源语义，并明确“源是什么意思”和“某后端怎样执行”不是同一件事。

## 10. PreparedDialogue 的理由

Prepared object 保存精确、可执行、可写入 metadata 的缓存计划。`HiggsPreparedTurn.__init__` 调用 `validate_higgs_plan()`；`CosyVoicePreparedTurn.__init__` 调用 `validate_cosyvoice_plan()`。只有验证成功的对象才会交给执行层。

计划在构造时 deep copy，`plan`/`rate_plan` 等属性也返回 defensive copy。这阻止下游意外修改冻结计划，避免实际请求与 metadata、QC 或 fingerprint 假设分叉。执行阶段不重新映射 controls。

## 11. Higgs 控制计划

Higgs 使用 frozen `map_turn_to_higgs()` / Controlled-TTS-v1 planner：

- arousal、affect、pause_within 与 hesitation 进入冻结的 model-input planning；
- 每 turn 固定一次 synthesis call；
- selected Higgs reference 在 preparation 和 preflight 都进行 live SHA 验证；
- `FROZEN_GENERATION_FIELDS` 固定 HTTP generation profile；
- Higgs model `speed` 始终为 `1.0`；
- production semantic rate 在模型之后处理：slow=`atempo=0.85`，normal=不处理，fast=`atempo=1.15`；
- `pause_before` 不进入 turn WAV，由 assembly 处理。

这些规则证明计划与请求一致，不证明 arousal 或 emotion 已经在听感上校准。

## 12. CosyVoice 控制计划

截至本文归档时，CosyVoice3 的映射为：

- rate：slow `0.8`、normal `1.0`、fast `1.2`；
- arousal/affect：provisional instruction mapping；
- hesitation：shared lexical planner，在 text 中规划；
- pause_before：assembly timing；
- pause_within：`0` 为 not required，`> 0` 明确 unsupported 并在 preflight fail closed；
- prompt WAV、prompt text、SHA、inference mode、instruction 都冻结在 prepared plan；
- zero_shot 与 instruct2 都显式使用 `text_frontend=False`。

该映射不声称 emotion fidelity、speaker similarity 或声学单调性成立。

## 13. Preflight 边界

`run_dialogue()` 在创建 output directory 和首次 synthesis request 前调用 `preflight_prepared_dialogue()`。它先检查 dialogue engine、turn ordinal、每个 turn backend，再调用后端 turn-local preflight。

目的不是省一次检查，而是确保后面的无效 turn、缺失 reference 或不支持 capability 不会在前几个 turn 已经消耗 GPU 并写出文件后才被发现。`synthesize_prepared_turns()` 假设 whole-dialogue preflight 已完成；backend `synthesize_prepared_turn()` 仍保留 turn-local defensive check。这是两层不同责任。

## 14. Higgs 运行时架构

```text
5703tts main process
  | JSON Lines over stdin/stdout
  v
higgs_worker.py
  | owns process group; server stdout/stderr -> worker stderr
  v
SGLang-Omni server
  | HTTP GET /health
  | HTTP POST /v1/audio/speech
  v
WAV response
```

Worker 隔离 server lifecycle，负责启动、health polling、复用、SIGTERM/SIGKILL 与 reap。父进程按 server executable、model dir、host、port、startup timeout、inference timeout 缓存一个 worker。stdout 只传协议；stderr 由后台线程持续 drain，因为无人读取的 pipe 填满后会阻塞 child。

请求错误在 server 仍存活时可报告为 recoverable；server 死亡、broken pipe 或 malformed protocol 是 fatal，父进程丢弃 worker cache。HTTP reference conditioning 的实际 shape 是：

```json
"references": [{"audio_path": "/resolved/reference.wav"}]
```

截至本文归档时，真实 Cloud 上的 SGLang/Higgs protocol、reference conditioning 和感知质量仍待验证。

## 15. CosyVoice 运行时架构

父进程使用 `config.tts.cosyvoice.python_bin` 启动 JSON-lines worker，因此 CosyVoice 依赖与主项目 `.venv` 隔离。Worker 把第三方 stdout 重定向到 stderr，加载一次模型后跨 turn 复用。每个 request 使用 prepared plan 的 text、prompt WAV/text、speed、mode 和可选 instruction。初始化失败结束 worker；per-request error 返回错误，让主进程决定后续 batch 行为。

## 16. Turn WAV 生命周期

Higgs worker 的实际生命周期是：

```text
turn_NNN.higgs_raw.wav.part
  -> atomic replace
turn_NNN.higgs_raw.wav
  -> normal: atomic rename
     or slow/fast: FFmpeg writes turn_NNN.higgs_processed.part.wav
  -> atomic replace
turn_NNN.wav
```

当前 failure boundary 只清理其负责的已知临时文件：rate postprocessing 前的 parent request/protocol failure 会删除 raw WAV 与 processed `.part`；但 `_apply_rate()` 内的 FFmpeg 或 processed-WAV failure 只保证删除 processed `.part`，已生成的 `turn_NNN.higgs_raw.wav` 可能保留。该 raw WAV 属于 pipeline-managed artifact，之后真正 rerender 时，Phase 2E cleanup 会在渲染前删除它。CosyVoice worker 直接保存最终 `turn_NNN.wav`。两种后端的最终 turn WAV 都只含 speech，不含 `pause_before`。

## 17. Assembly 与 timestamp

Assembly 要求 source/result ordinal 一一匹配、从 1 连续，并验证 source turn ID。先插入 `pause_before` silence，再记录 speech `start_sec`；因此 timestamp 覆盖 speech，不覆盖前导 pause。当前 none/short/long 是 0/500/900 ms，`pause_after_ms` 固定 0。

每个 speech segment 的短 fade 只防止硬切边缘 click；turn 之间没有 crossfade。

## 18. Telephone 输出

Telephone WAV 只是 clean dialogue 的：mono、resampling、high-pass、low-pass、level reduction。默认配置为 8 kHz、1 channel、300–3400 Hz、降低 3 dB。

它不是电话 codec，不包含 packet loss、line noise、room simulation，也不是完整 PSTN channel simulation。

## 19. Metadata 结构

`<dialogue_id>_metadata.json` 的用途是 provenance，而不是评分。顶层保存 dialogue/output/tts/provenance；每个 turn 分开保存：

- `source_identity`：ordinal、source turn ID、四种 role/speaker identity；
- `labels` 与 `requested`：源请求；
- `planned`：冻结 backend plan；
- `approved_speaker_reference`：实际 selected reference；
- `execution`：synthesis/rate status 与 turn filename；
- `timing`：pause 与 speech timestamp。

## 20. QC 范围

QC 是 structural/control-integrity QA：检查 dialogue/source SHA、turn count/order、source/metadata/plan 一致性、speaker/reference identity、execution status、timestamp、turn/clean/telephone WAV 可读取性。

QC 不证明 emotion accuracy、speaker similarity、naturalness、intelligibility 或 clinical validity。

## 21. Batch manifest

`batch_result.json` 的 `manifest_version` 固定为 `2.0`，是 output root 的 authoritative batch state。每个 result 的 action 是 `rendered`、`resumed`、`excluded_known_issue`、`input_error` 或 `render_failed`，并保存 source provenance、fingerprint、output directory、error/exclusion。

Manifest 通过 temporary file + replace 原子发布，但只在 batch 结束时写一次，不是每 dialogue checkpoint。

## 22. Fingerprint 与 resume 语义

每个可渲染 record 的 semantic fingerprint 包含：

- source `dialogue_id` + canonical `record_sha256`；
- shared render-affecting config：engine、fade、telephone；
- selected backend static identity；
- selected role/render speaker/reference materialization；
- exclusion decision（当前 eligible path 为 false）。

它故意不包含 input file location、JSON formatting、unselected backend config。Backend static identity 记录配置的 model/runtime 路径、control mapping 和 generation profile，但尚未对完整 checkpoint/runtime environment 做 cryptographic pin。

## 23. Artifact-integrity resume validation

Resume 必须同时满足：前次 result 成功且 fingerprint 相同；现有 artifacts 仍完整。后者高层检查：metadata 可严格解析、dialogue/source identity、backend/implementation identity、turn identity/order、labels/requested controls、timing 合理、预期文件名、turn/clean/telephone WAV 可读取、selected reference 当前 bytes 与 SHA 一致。

只存在几个 WAV 文件绝不够；任何完整性失败都会进入 rerender，而不是跳过。

## 24. Rerender stale cleanup

顺序固定为：valid resume -> 不 cleanup；否则 cleanup -> `run_dialogue()`。Cleanup 失败会阻止 render，避免在旧/新混合目录继续写入。

Cleanup 只删除 pipeline-owned direct-child 名称：三类 dialogue output 和合法 `turn_NNN`/Higgs intermediate 名称。不递归、不 `rmtree`、不广泛删除 `*.wav`。Unmanaged file 与 nested directory 保留；managed-name directory fail closed；managed child symlink 只 unlink link 本身，不 follow target。

`_safe_dialogue_output_path()` 要求 dialogue ID 只能命名 output root 的一个 direct child，拒绝 slash、backslash、absolute、`.`/`..` 与 per-dialogue directory symlink。Output root 本身可以是 symlink，但解析后仍须保持 direct-child 关系。

## 25. Exclusion handling

Known issue 来自显式 exclusion policy，不写死在 renderer。Policy 严格解析、排序并计算 `policy_sha256`。Exclusion 在 sidecar requirement 和 preparation 前决定，因此 excluded dialogue 不需要 reference、output directory 或 render fingerprint；manifest 仍记录原因、provenance 与 policy SHA。

## 26. Failure semantics

- JSONL parse 错误：单行 `input_error`，继续后续记录。
- final contract 错误：该 record 为 `input_error`，不适配 legacy schema。
- duplicate dialogue ID、unsafe output ID、malformed sidecar/manifest：batch-level fail before render。
- pipeline/preflight/synthesis/QC exception：`PipelineResult` 转换为该 dialogue 的 `render_failed`，其他 record 可继续。
- cleanup failure：该 dialogue `render_failed`，不进入 mixed-state rendering。
- 最终 exit code：batch 全 success 为 0，否则 1；CLI 初始化级异常直接抛出。

## 27. 重要安全边界

1. 严格 JSON parser 阻止 duplicate key 与非标准 numeric constant 造成歧义。
2. Final contract validator 是唯一生产 schema gate。
3. Sidecar source SHA 与 role/scenario identity 防止错误绑定。
4. Selected reference 的 path identity 与 live SHA 分开验证。
5. Whole-dialogue preflight 避免 partial GPU execution。
6. Output path 要求 direct child，拒绝 traversal/alias。
7. Cleanup 使用 managed-name allowlist，不相信 metadata 提供的任意路径。
8. Worker stdout 保留协议，stderr 持续 drain。

## 28. Reproducibility 与 provenance

当前已记录 source SHA、assignment/registry/active-speaker SHA、mapping/implementation identity、config SHA、backend runtime paths、selected reference SHA、requested/planned/executed/timing。它足以定位大部分软件语义变化。

限制是 backend identity 尚未 hash 完整 model checkpoint、Python packages、CUDA/driver 和 server binary bytes。真实 Higgs evidence package 必须另外记录这些环境身份。

## 29. 已知限制

- 截至本文归档时，main 尚未完成真实 Higgs Cloud validation。
- Higgs reference candidate 必须实际审核后才能成为 approved reference。
- 完整 model checkpoint/runtime environment 尚未 cryptographically pin。
- `batch_result.json` 在 batch 完成时写，而非每 dialogue checkpoint。
- Structural QC 不是 perceptual QA。
- Telephone processing 是 band-limit/resample，不是真实电话 codec。
- CosyVoice `pause_within > 0` 不支持并 fail closed。
- CosyVoice affect/arousal instruction mapping 仍为 provisional。
- 不存在 automatic backend fallback。
- 最终大规模 source corpus 未提交到 repository。

## 30. 截至本文归档时的生产准备状态

Software/offline path 已强验证。Phase 3A 留有真实 CosyVoice mini-batch evidence：5 dialogues、30 turns、0 failures、identical resume 5/5，并通过 controlled corruption rerender 与 stale-turn cleanup 检查。这是 CosyVoice runtime evidence，不是 Higgs evidence，也不证明感知 fidelity。

截至本文归档时，Higgs production integration 和 offline validation 已完成，但 reference-conditioned Cloud runtime、speaker separation、perceptual review、resource stability 与 performance gate 尚未通过。因此不能称 Higgs 已 production-ready。最终大语料 handoff 后仍需要 large-scale production rehearsal。

## 31. Higgs Cloud 下一步

以合并后的 exact `main` SHA 为基线，按 [`HIGGS_CLOUD_VALIDATION.md`](../HIGGS_CLOUD_VALIDATION.md) 的 evidence checklist 和 [`HIGGS_PRODUCTION_RUNBOOK_CN.md`](../HIGGS_PRODUCTION_RUNBOOK_CN.md) 的执行步骤：准备 runtime/model，审核 Higgs reference candidate，生成 review-only sidecar，跑 startup、单 turn、control matrix、双 speaker、小 batch、resume 和 shutdown/resource 检查，保存配置/环境/音频/metadata/QC/manifest evidence，最后由人工 reviewer 决定 production approval。

## 32. 按文件阅读地图

| 文件 | 主要问题 |
| --- | --- |
| `input/records.py` | bytes 如何变成稳定 source record？ |
| `input/contract.py`、`controlled_tts/schema.py` | final schema 与 normalized semantics 是什么？ |
| `input/exclusions.py` | known issue 如何数据化？ |
| `speaker_references.py` | selected reference 如何解析/验 SHA？ |
| `render_models.py` | canonical/prepared/result 各自拥有什么？ |
| `render_plan.py`、`plan_validation.py` | source 如何成为冻结 backend plan？ |
| `tts_engine.py` | whole-dialogue preflight 与顺序执行在哪里？ |
| `backends/higgs*.py` | parent/worker/SGLang、HTTP 与 atempo 生命周期 |
| `backends/cosyvoice*.py` | 独立 Python worker 与 provisional control mapping |
| `render/assemble.py` | pause、fade、timestamp 如何形成？ |
| `render/postprocess.py` | telephone output 实际做什么？ |
| `render/metadata.py`、`render/qc.py` | provenance 与 structural QA 边界 |
| `pipeline.py` | 单 dialogue orchestration |
| `batch_identity.py`、`batch.py` | fingerprint、resume、cleanup、manifest |
| `cli.py`、`config.py` | CLI discovery 与显式 backend selection |
| `scripts/materialize_speaker_assignments.py` | sidecar 如何产生且不改源记录？ |

## 33. 示例命令

基础检查：

```bash
uv run --with pytest pytest -q
uv run ruff check .
git diff --check
```

生成双后端 sidecar：

```bash
uv run python scripts/materialize_speaker_assignments.py \
  --input data/final/dialogues.jsonl \
  --assignments data/speaker_assignments.jsonl \
  --registry data/speaker_pool/vctk_v0.1/speaker_registry.json \
  --active-speakers data/speaker_pool/vctk_v0.1/active_speakers.json \
  --manifest data/speaker_sidecar.json \
  --backend both
```

Higgs production-primary run：

```bash
uv run 5703tts \
  --input data/final/dialogues.jsonl \
  --speaker-sidecar data/speaker_sidecar.json \
  --exclusion-policy data/exclusions.json \
  --config config/config.yaml \
  --output data/output \
  --resume
```

CosyVoice backup run 必须显式换配置：

```bash
uv run 5703tts \
  --input data/final/dialogues.jsonl \
  --speaker-sidecar data/speaker_sidecar.json \
  --config config/config_cosyvoice.yaml \
  --output data/output_cv3
```

## 34. Troubleshooting 入口

- 输入 parse/duplicate key/row：`input/records.py` 与 manifest 的 `input_error`。
- final schema/control domain：`input/contract.py`、`controlled_tts/schema.py`。
- sidecar role/source SHA/reference：`render_plan.py`、`speaker_references.py`。
- 在 output directory 创建前失败：查看 whole-dialogue preflight 和 backend turn preflight。
- Higgs startup/HTTP/shutdown：主 log 的 `higgs_worker_stderr`，再对照 `higgs.py`/`higgs_worker.py`。
- CosyVoice dependency/model：`python_bin`、`repo_dir`、`model_dir` 与 worker stderr。
- resume 未命中：比较 fingerprint、metadata identity、WAV readability 与 live reference SHA。
- rerender cleanup 失败：检查 managed-name path 是否变成 directory 或 dialogue directory 是否为 symlink。
- QC failed：查看 metadata 中 requested/planned/execution/timing 的分层，不把 structural failure 当作听感结论。
- Higgs Cloud gate：以 [`HIGGS_CLOUD_VALIDATION.md`](../HIGGS_CLOUD_VALIDATION.md) 为正式 checklist，以 [`HIGGS_PRODUCTION_RUNBOOK_CN.md`](../HIGGS_PRODUCTION_RUNBOOK_CN.md) 为执行步骤。
