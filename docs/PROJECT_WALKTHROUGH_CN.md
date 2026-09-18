# 5703tts 生产路径导读

## 当前产品边界

生产入口只有一条：`upstream/Controlled-TTS-v1` 交付的 final nested JSON/JSONL。
普通 `5703tts` 命令不再根据 schema family 选择 renderer，也不接受 legacy flat
或 schema v0.2。

```text
final JSON/JSONL
  -> InputRecord
  -> final validation
  -> exclusion policy
  -> speaker sidecar
  -> CanonicalDialogue
  -> Higgs plan 或 CosyVoice3 plan
  -> TurnRenderResult
  -> shared assembly / metadata / QC
  -> manifest v2 / resume
```

Higgs 是 primary production backend；CosyVoice3 是 backup/secondary。切换必须显式
修改 `tts.engine`，整个 batch 只使用一个 backend。失败后绝不自动 fallback。

## 输入与 validation

`.json` 包含一个 final dialogue；`.jsonl` 每个非空行包含一个 dialogue；目录会发现
这两种 container。读取层拒绝 duplicate JSON keys、NaN/Infinity、非对象记录和空
`dialogue_id`。JSONL 单行 parse 错误记录为 `input_error`，后续有效记录继续。

所有成功 parse 的记录直接交给 `validate_final_dialogue()`。Legacy/v0.2 会得到清晰的
production-contract 错误，不会被重新解释成 `NormalizedTurn`。跨 container 的重复
`dialogue_id` 与不安全输出 ID 会在 render 前失败。

## Canonical 与 backend plan

`CanonicalTurn` 保存 source text、labels、upstream/logical role、scenario speaker、
production `spk_*`、raw request，以及 normalized rate/arousal/affect/pause/hesitation。
它不保存 Higgs model input、Higgs token、CosyVoice instruction 或 speed。

Higgs 继续使用 frozen `map_turn_to_higgs()`；0.85/1.15 atempo、pause token、hesitation、
reference 与 generation profile 语义不变。CosyVoice plan 缓存 0.8/1.0/1.2 speed、
instruction、prompt WAV/text/SHA 和 capability 状态。执行、metadata、QC 都只消费缓存
plan，不重新调用 planner。

CosyVoice 对 `pause_within > 0` 没有 evidence-backed deterministic realization，因此在
创建 output directory 之前 fail closed。`pause_before` 仍由 assembly 按 0/500/900 ms
实现；turn WAV 只包含 speech。

## Speaker sidecar

Final source 永不改写。Sidecar 为 caller/counsellor 保存 upstream identity、scenario
speaker、production `spk_*` 以及所请求 backend 的 reference。

Materializer 必须显式选择：

- `--backend higgs`：只要求/输出 approved `higgs_reference`。
- `--backend cosyvoice`：只要求/输出 registry `primary_reference`，保留 exact
  `prompt_text`。
- `--backend both`：两者都要求并输出。

两种 reference 绝不互相替代；assignment 与 target 无关。

## Config 与运行

`config/config.yaml` 是 dual-backend production template，默认 `tts.engine: higgs`。
其中 Higgs 路径是明确的 external-asset placeholder，真实运行前必须 provision。
CosyVoice block 同时存在，但只有显式选择 `engine: cosyvoice` 才参与运行。未选择的
backend 缺失或无效不会阻止选中 backend。

```bash
uv run 5703tts \
  --input data/final/dialogues.jsonl \
  --speaker-sidecar data/speaker_sidecar.json \
  --config config/config.yaml \
  --output data/output \
  --resume
```

## Resume、manifest 与 exclusion

Manifest v2 是唯一 production manifest。Fingerprint 包含 canonical source SHA、shared
audio settings、selected backend semantic/runtime identity，以及 selected role/reference
identity；不包含 unselected backend、JSONL line number 或 YAML formatting。

Resume 接受旧结果前会重新读取选中 backend 的 WAV，并比较 live SHA-256。文件缺失、
不可读或 bytes 改变都会进入正常 prepare/preflight，而不是 skip。Planner 不参与 resume。

Exclusion 在 speaker lookup/preparation 前执行。Excluded record 计入输入总数，但不需要
reference、fingerprint 或 output directory；known issue 必须来自 policy data，不能写死。

## 仍待 U3 删除的内容

`validate.py`、`schemas/dialogue_schema.json`、legacy normalized pipeline、Kokoro、旧
benchmark/compatibility tests 仍物理存在，但普通 production CLI 不可达。本阶段不批量
删除它们。

## Evidence 边界

Offline suite 证明 structural/control integrity、cached request、ordering、timing、metadata、
QC 与 resume，不证明 emotion、arousal、speaker similarity、intelligibility 或 naturalness。
下一 runtime gate 是真实 Higgs cloud/GPU validation；本阶段没有运行模型、worker、网络
或下载。
