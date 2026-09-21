# Fresh Ubuntu GPU server 安装

本指南分为三层：

1. Base environment
2. CosyVoice3 backup
3. Higgs3 primary

本流程假定 Google Cloud VM 已关闭 Secure Boot；Part 1 会检查，若仍为 enabled 必须停止并重建或重新配置 VM。磁盘容量是 operational guidance：fresh Cloud VM 建议至少约 100 GB，150 GB 可为两个约 9–10 GB models、三套 Python environments、CUDA/SGLang packages、Hugging Face cache/temp、UCX build 和 validation outputs 提供更安全余量。

## 自动安装（推荐）

Clone 后运行 resumable bootstrap；若 NVIDIA/CUDA 安装触发 reboot，SSH 重连、回到同一 repository，再执行同一 command。

```bash
bash scripts/setup_cloud_environment.sh
```

只读诊断使用 `bash scripts/setup_cloud_environment.sh --check`。下面 Part 1 / Part 2 / Part 3 的 manual commands 仍是 authoritative troubleshooting/reference path。

# Part 1 — 基础环境

## 1. 检查 Ubuntu

当前 Google Cloud GPU 安装路径面向 Ubuntu 22.04 LTS 或 Ubuntu 24.04 LTS。

```bash
cat /etc/os-release
uname -a
```

关键检查：`VERSION_ID` 为 `22.04` 或 `24.04`。

## 2. 安装基础系统组件

`git` 用于 checkout；`curl/wget` 用于 installer/download；FFmpeg、SoX/libsndfile 提供音频支持；build-essential/pkg-config 供 native dependency 编译；`iproute2` 提供后续使用的 `ss`；`mokutil` 检查 Secure Boot。本步骤不安装 model-specific libraries。

```bash
set -euo pipefail

sudo apt-get update
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git \
  curl \
  wget \
  ca-certificates \
  python3 \
  ffmpeg \
  sox \
  libsndfile1 \
  build-essential \
  pkg-config \
  iproute2 \
  mokutil

git --version
curl --version | head -n 1
python3 --version
ffmpeg -version | head -n 1
sox --version
ss --version || ss -h >/dev/null

SECURE_BOOT_STATE="$(mokutil --sb-state)"
printf '%s\n' "$SECURE_BOOT_STATE"
if ! grep -qi '^SecureBoot disabled$' <<<"$SECURE_BOOT_STATE"; then
  echo 'ERROR: this installer path requires Secure Boot disabled' >&2
  exit 1
fi
```

关键检查：system tools 均可用且输出 `SecureBoot disabled`；若不是 disabled，STOP，不要自动修改 firmware/security settings。

## 3. 检查 / 安装 NVIDIA driver

若 `nvidia-smi` 已工作，本 block 会跳过安装；否则按 [Google Cloud 官方说明](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu) 安装 production driver、reboot，SSH 重连后重新执行本 block。Part 1 只建立 driver；`nvidia-smi` 的 “CUDA Version” 是 driver capability，不代表已安装 `/usr/local/cuda` toolkit，CUDA Toolkit 属于 Part 3。

```bash
set -euo pipefail

if nvidia-smi; then
  echo 'NVIDIA driver already works; installation skipped.'
else
  cd "$HOME"
  curl -fL \
    https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz \
    -o cuda_installer.pyz
  sudo systemctl stop google-cloud-ops-agent 2>/dev/null || true
  sudo python3 "$HOME/cuda_installer.pyz" install_driver \
    --installation-mode=repo \
    --installation-branch=prod
  echo 'Driver installed; rebooting now. Reconnect with SSH and rerun this block.'
  sudo reboot
fi

nvidia-smi
echo 'NVIDIA_DRIVER_OK=1'
```

关键检查：reboot 后显示 GPU/driver 表，并输出 `NVIDIA_DRIVER_OK=1`；本步骤没有安装 CUDA Toolkit。

## 4. 安装 uv

使用 [Astral 官方 installer](https://docs.astral.sh/uv/getting-started/installation/) 并把 `$HOME/.local/bin` 写入登录 PATH。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

export PATH="$HOME/.local/bin:$PATH"
uv --version
command -v uv

grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.profile" || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.profile"
source "$HOME/.profile"
uv --version
```

关键检查：`uv --version` 成功，`command -v uv` 指向 `$HOME/.local/bin/uv`。

## 5. 安装 Python 3.11

Repository 的 `.python-version` 是 `3.11`；Part 1 只安装 Python 3.11，backend-specific Python 留到后续阶段。

```bash
uv python install 3.11
uv python find 3.11
"$(uv python find 3.11)" --version
```

关键检查：输出 `Python 3.11.x`；本步骤不安装 Python 3.10 或 3.12。

## 6. Clone 5703tts

当前 documentation-development baseline 是下列 SHA；Phase 4B 提交后，Cloud validation 前必须把它替换为最终 Phase 4B commit，不能预填未来 SHA。

```bash
cd "$HOME"
git clone https://github.com/Li-PengSheng/5703tts.git
cd "$HOME/5703tts"
git checkout --detach \
  969f52a4bebc9dcf17685a10135b7ad1672e1c1e
git rev-parse HEAD
git status --short
```

关键检查：HEAD 为 `969f52a4bebc9dcf17685a10135b7ad1672e1c1e`，fresh checkout 的 status 为空。

## 7. 创建项目 .venv

使用 committed `uv.lock` 创建 project environment；不要手工 `pip install` project dependencies。

```bash
cd "$HOME/5703tts"
uv venv --python 3.11 .venv
uv sync --frozen
uv run python --version
uv run 5703tts --help
```

关键检查：project Python 为 `3.11.x`，`5703tts --help` 正常输出。

## 8. 验证基础项目环境

运行 repository 的静态检查和 offline tests；本步骤不使用 model、backend runtime 或 GPU inference。

```bash
cd "$HOME/5703tts"
uv run ruff check .
uv run --with pytest pytest -q
```

关键检查：Ruff 通过，pytest 完成。只有 optional `comp5703-tts-experiments` checkout 存在时可能出现已知 local parity mismatch；它不是安装失败，也不要修改 tests 来隐藏它。

## 9. 基础环境最终检查

此 block 只在所有 mandatory checks 成功后打印 success marker。

```bash
set -euo pipefail

cd "$HOME/5703tts"

echo '=== OS ==='
. /etc/os-release
echo "$PRETTY_NAME"

echo '=== SYSTEM ==='
git --version
curl --version | head -n 1
ffmpeg -version | head -n 1
sox --version

echo '=== GPU DRIVER ==='
nvidia-smi

echo '=== UV ==='
uv --version

echo '=== PROJECT PYTHON ==='
uv run python --version

echo '=== 5703TTS CLI ==='
uv run 5703tts --help >/dev/null

echo 'BASE_ENVIRONMENT_OK=1'
```

关键检查：最后一行是 `BASE_ENVIRONMENT_OK=1`；任何此前命令失败时都不会打印它。

Part 1 完成后的 repository 目录应包含：

```text
5703tts/
├── .venv/
├── config/
├── data/
├── docs/
├── scripts/
├── src/
├── tests/
├── pyproject.toml
└── uv.lock
```

`models/` 和 `third_party/` 此时可以不存在或为空。Part 1 尚未创建 `third_party/CosyVoice`、`third_party/sglang-omni`、`models/Fun-CosyVoice3-0.5B` 或 `models/higgs-tts-3-4b`。

# Part 2 — CosyVoice3 Backup

本节按 [CosyVoice 官方 repository](https://github.com/FunAudioLLM/CosyVoice) 和 [Fun-CosyVoice3-0.5B-2512 model](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) 安装 backup runtime，不执行 production synthesis。

## 1. 安装独立 Python 3.10

CosyVoice 使用独立 Python 3.10；不要改变 `5703tts/.venv`。

```bash
cd "$HOME/5703tts"
uv python install 3.10
uv python find 3.10
"$(uv python find 3.10)" --version
```

关键检查：输出 `Python 3.10.x`，project `.venv` 仍使用 Part 1 的 Python 3.11。

## 2. Clone 并固定 CosyVoice source

为可复现 Cloud setup，本指南将 upstream CosyVoice source 固定在 `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`，其 Matcha-TTS gitlink 为 `dd9105b34bf2be2230f4aa1e4769fb586a3c824e`，不跟随 moving `main`。Recursive submodule 是 mandatory，因为 worker 要求 `third_party/CosyVoice/third_party/Matcha-TTS`。

```bash
cd "$HOME/5703tts"
mkdir -p third_party
git clone --recursive \
  https://github.com/FunAudioLLM/CosyVoice.git \
  third_party/CosyVoice
git -C third_party/CosyVoice checkout --detach \
  074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
git -C third_party/CosyVoice submodule sync --recursive
git -C third_party/CosyVoice submodule update \
  --init \
  --recursive
git -C third_party/CosyVoice rev-parse HEAD
git -C third_party/CosyVoice status --short
test -d third_party/CosyVoice/third_party/Matcha-TTS
git -C third_party/CosyVoice/third_party/Matcha-TTS rev-parse HEAD
echo 'COSYVOICE_SOURCE_OK=1'
```

关键检查：CosyVoice 输出 `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`，Matcha-TTS 输出 `dd9105b34bf2be2230f4aa1e4769fb586a3c824e`，source status 为空，最后输出 `COSYVOICE_SOURCE_OK=1`。

## 3. 创建隔离 CosyVoice .venv

必须使用 project config 已指定的 `third_party/CosyVoice/.venv`。

```bash
cd "$HOME/5703tts"
uv venv \
  --python 3.10 \
  third_party/CosyVoice/.venv
third_party/CosyVoice/.venv/bin/python --version
```

关键检查：输出 `Python 3.10.x`，interpreter 位于 `third_party/CosyVoice/.venv/bin/python`。

## 4. 安装官方 requirements

直接安装 pinned checkout 的 `requirements.txt`，不手工重建 dependency list；它管理自己的 Torch/CUDA-related packages，不要在此环境安装 Higgs Torch stack 或 SGLang。

```bash
cd "$HOME/5703tts"
uv pip install \
  --python third_party/CosyVoice/.venv/bin/python \
  -r third_party/CosyVoice/requirements.txt

third_party/CosyVoice/.venv/bin/python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for package in (
    "torch",
    "torchaudio",
    "transformers",
    "onnxruntime-gpu",
    "deepspeed",
):
    try:
        print(f"{package}={version(package)}")
    except PackageNotFoundError:
        print(f"{package}=NOT_INSTALLED")
PY
```

关键检查：保存实际 resolved versions；不要用文档中的猜测版本替代此输出，也不要修改 project `.venv`。

## 5. 验证 PyTorch GPU access

CosyVoice environment 必须能使用 CUDA；本检查失败时不要继续 model validation。

```bash
cd "$HOME/5703tts"
third_party/CosyVoice/.venv/bin/python - <<'PY'
import torch

available = torch.cuda.is_available()

print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.is_available()={available}")

if not available:
    raise SystemExit("ERROR: CosyVoice Python environment cannot access CUDA")

print(f"GPU={torch.cuda.get_device_name(0)}")
PY
```

关键检查：`torch.cuda.is_available()=True`，并输出实际 Torch、CUDA build marker 和 GPU name。

## 6. 安装 Hugging Face CLI

把 CLI 安装为 user-level uv tool，不放进 project 或 CosyVoice environment；public model 正常下载不要求 login。

```bash
export PATH="$HOME/.local/bin:$PATH"
if ! command -v hf >/dev/null 2>&1; then
  uv tool install "huggingface_hub[hf_xet]"
fi
hf version
```

关键检查：`hf version` 成功。只有下载报告 authentication/rate-limit 问题时才运行 `hf auth login` 和 `hf auth whoami`；不要把 token 写入 YAML、shell script、Git 或 log。

## 7. 下载固定 CV3 model

Remote model 是 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`；local directory 故意使用 config 要求的 `models/Fun-CosyVoice3-0.5B`。Current Cloud installation pin 是 revision `29e01c4e8d000f4bcd70751be16fa94bf3d85a18`。

```bash
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/5703tts"
mkdir -p models/Fun-CosyVoice3-0.5B
hf download \
  FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --revision 29e01c4e8d000f4bcd70751be16fa94bf3d85a18 \
  --local-dir models/Fun-CosyVoice3-0.5B
printf '%s\n' \
  'remote_model=FunAudioLLM/Fun-CosyVoice3-0.5B-2512' \
  'local_directory=models/Fun-CosyVoice3-0.5B' \
  'downloaded_revision=29e01c4e8d000f4bcd70751be16fa94bf3d85a18'
find models/Fun-CosyVoice3-0.5B/.cache/huggingface/trees \
  -maxdepth 1 -type f -printf '%f\n' | sort
```

关键检查：保留 `hf download` 输出和三行 identity 作为 new Cloud evidence；metadata tree 包含 `29e01c4e8d000f4bcd70751be16fa94bf3d85a18.json`。

## 8. 验证 mandatory model files

Model 约 9.75 GB，但 size 不是 gate；worker 至少要求 `cosyvoice3.yaml`，安装 gate 还检查三个主要 weights。

```bash
cd "$HOME/5703tts"
test -f models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml
test -f models/Fun-CosyVoice3-0.5B/llm.pt
test -f models/Fun-CosyVoice3-0.5B/flow.pt
test -f models/Fun-CosyVoice3-0.5B/hift.pt
du -sh models/Fun-CosyVoice3-0.5B
echo 'COSYVOICE_MODEL_FILES_OK=1'
```

关键检查：四个 files 存在，最后输出 `COSYVOICE_MODEL_FILES_OK=1`；不要要求 arbitrary exact directory size。

## 9. 验证 import 与 text frontend boundary

Worker 会临时加入 CosyVoice 和 Matcha-TTS paths，不需要永久设置 `PYTHONPATH`。Production calls 固定 `text_frontend=False`，所以 upstream `CosyVoice-ttsfrd` 不是 mandatory；constructor 仍可能检查 frontend/WeText availability 或下载相关资源。

```bash
cd "$HOME/5703tts"
third_party/CosyVoice/.venv/bin/python - <<'PY'
import sys
from pathlib import Path

root = Path("third_party/CosyVoice").resolve()
matcha = root / "third_party" / "Matcha-TTS"

sys.path.insert(0, str(matcha))
sys.path.insert(0, str(root))

from cosyvoice.cli.cosyvoice import CosyVoice3

print("CosyVoice3 import OK")
PY
```

关键检查：输出 `CosyVoice3 import OK`；不要向 `~/.profile` 添加 CosyVoice `PYTHONPATH`，也不要把 optional `ttsfrd` 当作当前 production requirement。

## 10. 验证 checked-in backup config

不要创建另一份 CV3 config；按下列 paths 安装后，`config/config_cosyvoice.yaml` 无需修改，validated defaults 保持 `load_trt: false`、`load_vllm: false`、`fp16: true`。

```bash
cd "$HOME/5703tts"
sed -n '1,80p' config/config_cosyvoice.yaml
grep -F 'python_bin: third_party/CosyVoice/.venv/bin/python' \
  config/config_cosyvoice.yaml
grep -F 'repo_dir: third_party/CosyVoice' config/config_cosyvoice.yaml
grep -F 'model_dir: models/Fun-CosyVoice3-0.5B' config/config_cosyvoice.yaml
grep -F 'load_trt: false' config/config_cosyvoice.yaml
grep -F 'load_vllm: false' config/config_cosyvoice.yaml
grep -F 'fp16: true' config/config_cosyvoice.yaml
test -x third_party/CosyVoice/.venv/bin/python
test -d third_party/CosyVoice
test -d third_party/CosyVoice/third_party/Matcha-TTS
test -d models/Fun-CosyVoice3-0.5B
test -f models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml
echo 'COSYVOICE_CONFIG_PATHS_OK=1'
```

关键检查：所有 checked-in values 和 paths 匹配，最后输出 `COSYVOICE_CONFIG_PATHS_OK=1`；不要添加 TensorRT/vLLM deployment instructions 或改变 defaults。

## 11. Optional：constructor-only GPU check

此检查与 worker construction 一致，可能下载 frontend/WeText resource、消耗大量时间和 VRAM；它只构造 model，不 synthesis。若 Cloud 安装阶段不允许额外 frontend download，跳过并在 production validation runbook 中验证完整 startup。

```bash
cd "$HOME/5703tts"
read -r -p 'Run optional GPU-heavy CosyVoice3 constructor check? [y/N] ' answer
if [[ "$answer" == 'y' || "$answer" == 'Y' ]]; then
  third_party/CosyVoice/.venv/bin/python - <<'PY'
import sys
from pathlib import Path

root = Path("third_party/CosyVoice").resolve()
matcha = root / "third_party" / "Matcha-TTS"
model_dir = Path("models/Fun-CosyVoice3-0.5B").resolve()

sys.path.insert(0, str(matcha))
sys.path.insert(0, str(root))

from cosyvoice.cli.cosyvoice import CosyVoice3

model = CosyVoice3(
    model_dir=str(model_dir),
    load_trt=False,
    load_vllm=False,
    fp16=True,
)
print(f"CosyVoice3 constructor OK; sample_rate={model.sample_rate}")
PY
else
  echo 'OPTIONAL_CONSTRUCTOR_CHECK=SKIPPED'
fi
```

关键检查：选择执行时输出 `CosyVoice3 constructor OK`；本步骤不调用 inference、pipeline render、TensorRT 或 vLLM。

## 12. CosyVoice backup 最终检查

此 mandatory gate 不 synthesis；任何检查失败时都不会打印 success marker。

```bash
set -euo pipefail

cd "$HOME/5703tts"

echo '=== COSYVOICE SOURCE ==='
test -d third_party/CosyVoice
test -d third_party/CosyVoice/third_party/Matcha-TTS
git -C third_party/CosyVoice rev-parse HEAD

echo '=== COSYVOICE PYTHON ==='
test -x third_party/CosyVoice/.venv/bin/python
third_party/CosyVoice/.venv/bin/python --version

echo '=== COSYVOICE GPU ==='
third_party/CosyVoice/.venv/bin/python - <<'PY'
import torch

assert torch.cuda.is_available(), "CosyVoice CUDA unavailable"
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

echo '=== COSYVOICE MODEL ==='
test -f models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml
test -f models/Fun-CosyVoice3-0.5B/llm.pt
test -f models/Fun-CosyVoice3-0.5B/flow.pt
test -f models/Fun-CosyVoice3-0.5B/hift.pt

echo '=== PROJECT CONFIG ==='
test -f config/config_cosyvoice.yaml

echo 'COSYVOICE_BACKUP_ENVIRONMENT_OK=1'
```

关键检查：最后输出 `COSYVOICE_BACKUP_ENVIRONMENT_OK=1`。Production synthesis、speaker references 和 batch render 留给 runtime validation。

Part 2 完成后的新增目录应为：

```text
5703tts/
├── third_party/
│   └── CosyVoice/
│       ├── .venv/
│       ├── cosyvoice/
│       └── third_party/
│           └── Matcha-TTS/
└── models/
    └── Fun-CosyVoice3-0.5B/
        ├── cosyvoice3.yaml
        ├── llm.pt
        ├── flow.pt
        └── hift.pt
```

# Part 3 — Higgs3 Primary

首次 Cloud validation 固定 `sglang-omni==0.1.3` 和 `bosonai/higgs-tts-3-4b@0056125158f940389ab0808a581b8b2c590b32d4`，复现项目既有 evidence-compatible runtime；当前 upstream 已有更新版本和 Higgs 命名，但升级必须作为新的 evidence set。官方依据：[Google Cloud CUDA installer](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu)、[SGLang-Omni v0.1.3 installation](https://github.com/sgl-project/sglang-omni/blob/v0.1.3/docs/get_started/installation.md)、[v0.1.3 package](https://pypi.org/project/sglang-omni/0.1.3/)、[v0.1.3 Dockerfile](https://github.com/sgl-project/sglang-omni/blob/v0.1.3/docker/Dockerfile) 和 [frozen Higgs checkpoint](https://huggingface.co/bosonai/higgs-tts-3-4b/tree/0056125158f940389ab0808a581b8b2c590b32d4)。

## 1. 安装 CUDA Toolkit

Part 1 只安装了 NVIDIA driver；这里用 Google 官方 installer 安装 Toolkit。Installer 可能在完成前 reboot：SSH 重连后必须重新执行下面同一 block，重复至 installer 明确成功；成功返回后 block 会执行最后一次 reboot，再次重连后才进入第 2 节。

```bash
cd "$HOME"

if [ ! -f "$HOME/cuda_installer.pyz" ]; then
  curl -fL \
    https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz \
    -o "$HOME/cuda_installer.pyz"
fi

sudo python3 "$HOME/cuda_installer.pyz" install_cuda \
  --installation-mode=repo \
  --installation-branch=prod

sudo reboot
```

关键检查：顺序是 run block → 若 reboot/提示 rerun，SSH reconnect → run same block → repeat until installer success → final reboot → reconnect 后验证；不自动跨 reboot 循环，也不以 `|| true` 隐藏失败。

## 2. 重连并验证 CUDA 13 Toolkit

`nvidia-smi` 的 “CUDA Version” 只表示 driver capability；以 `/usr/local/cuda/bin/nvcc` 为 Toolkit gate。UCX 会针对 `/usr/local/cuda` 编译，且 pinned SGLang-Omni 0.1.3 dependency family 面向 CUDA 13；若 `nvcc` 不是 13.x，立即停止并记录 mismatch。

```bash
set -euo pipefail

nvidia-smi
test -d /usr/local/cuda
test -x /usr/local/cuda/bin/nvcc

NVCC_OUTPUT="$(/usr/local/cuda/bin/nvcc --version)"
printf '%s\n' "$NVCC_OUTPUT"

if ! grep -Eq 'release 13\.' <<<"$NVCC_OUTPUT"; then
  echo 'ERROR: first Cloud validation requires a CUDA 13.x Toolkit' >&2
  exit 1
fi

echo 'CUDA_TOOLKIT_13_OK=1'
```

关键检查：保存完整 `nvcc --version` 输出，最后必须出现 `CUDA_TOOLKIT_13_OK=1`。

## 3. 安装 UCX build dependencies

这些 packages 仅供 Higgs/SGLang 的 CUDA + verbs UCX build，不回填 Part 1。

```bash
sudo apt-get update

sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
  autoconf \
  automake \
  libtool \
  make \
  gcc \
  g++ \
  flex \
  bison \
  m4 \
  libnuma-dev \
  libibverbs-dev \
  librdmacm-dev \
  rdma-core \
  libnl-3-dev \
  libnl-route-3-dev \
  libudev-dev
```

关键检查：APT command 成功退出；不要把这些 backend-specific build dependencies 放入项目 `.venv`。

## 4. Build pinned UCX

使用 SGLang-Omni v0.1.3 Dockerfile 固定的 UCX 1.20.x commit 和 build flags；保留 source directory 供首次 validation evidence/troubleshooting。

```bash
set -euo pipefail

cd "$HOME"
mkdir -p "$HOME/src"
rm -rf "$HOME/src/ucx"

git clone --filter=blob:none \
  https://github.com/openucx/ucx.git \
  "$HOME/src/ucx"

git -C "$HOME/src/ucx" checkout \
  d8e50df6651b9ea5b76f23aee0aefbf053a4137a

cd "$HOME/src/ucx"
./autogen.sh

./contrib/configure-release-mt \
  --enable-shared \
  --disable-static \
  --disable-doxygen-doc \
  --enable-optimizations \
  --enable-cma \
  --enable-devel-headers \
  --with-cuda=/usr/local/cuda \
  --with-verbs \
  --with-dm \
  --prefix=/usr/local

make -j"$(nproc)"
sudo make install-strip
sudo ldconfig

command -v ucx_info
ucx_info -v
ucx_info -d | grep -i cuda || true

if ! ucx_info -d | grep -qi cuda; then
  echo 'ERROR: UCX has no visible CUDA component; inspect configure/build output' >&2
  exit 1
fi

echo 'UCX_CUDA_VERBS_OK=1'
```

关键检查：`ucx_info -v` 成功且 CUDA component 可见；否则停止并检查 configure/build output。

## 5. 安装 Python 3.12

Python 3.12 只服务 Higgs runtime；不得改变项目 Python 3.11 `.venv` 或 CosyVoice Python 3.10 `.venv`。

```bash
uv python install 3.12
uv python find 3.12
"$(uv python find 3.12)" --version
```

关键检查：输出 `Python 3.12.x`。

## 6. 创建 isolated SGLang-Omni environment

在 project-local ignored `third_party` path 创建第三套环境，它不是 5703tts project `.venv`。

```bash
cd "$HOME/5703tts"
mkdir -p third_party/sglang-omni

uv venv \
  --python 3.12 \
  third_party/sglang-omni/.venv

third_party/sglang-omni/.venv/bin/python --version
```

关键检查：runtime Python 为 `3.12.x`，path 为 `third_party/sglang-omni/.venv`。

## 7. 安装 SGLang-Omni 0.1.3

安装首次 validation pin，不安装 current latest，也不手工 override Torch/SGLang/Transformers dependencies。

```bash
cd "$HOME/5703tts"

uv pip install \
  --python third_party/sglang-omni/.venv/bin/python \
  --prerelease=allow \
  "sglang-omni==0.1.3"
```

关键检查：resolver 成功安装 `sglang-omni==0.1.3`；package 自身的 exact pins 是 Python dependency source of truth。

## 8. 验证 production executable

当前 Higgs preflight 要求 `server_executable` 是真实、可执行的 file；这里只检查 CLI，不启动 server。

```bash
cd "$HOME/5703tts"

test -x third_party/sglang-omni/.venv/bin/sgl-omni
third_party/sglang-omni/.venv/bin/sgl-omni --help
realpath third_party/sglang-omni/.venv/bin/sgl-omni

echo 'SGLANG_EXECUTABLE_OK=1'
```

关键检查：打印 absolute executable path，最后输出 `SGLANG_EXECUTABLE_OK=1`；不要运行 `sgl-omni serve`。

## 9. 记录 resolved package identity

打印实际 resolved versions，并对 v0.1.3 的 exact pins 做 gate；不伪造未安装 package 的版本。

```bash
cd "$HOME/5703tts"

third_party/sglang-omni/.venv/bin/python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

packages = (
    "sglang-omni",
    "sglang",
    "torch",
    "transformers",
    "flashinfer-python",
    "flash-attn-4",
)
expected = {
    "sglang-omni": "0.1.3",
    "sglang": "0.5.16",
    "torch": "2.11.0",
    "transformers": "5.12.1",
}

resolved = {}
for package in packages:
    try:
        resolved[package] = version(package)
    except PackageNotFoundError:
        resolved[package] = "NOT_INSTALLED"
    print(f"{package}={resolved[package]}")

mismatches = {
    package: (expected_version, resolved[package])
    for package, expected_version in expected.items()
    if resolved[package] != expected_version
}
if mismatches:
    raise SystemExit(f"ERROR: pinned dependency mismatch: {mismatches}")
PY
```

关键检查：mandatory identities 必须是 `0.1.3 / 0.5.16 / 2.11.0 / 5.12.1`；任何 mismatch 都先停止调查。

## 10. 验证 PyTorch CUDA access

记录实际 Torch CUDA build marker 和 GPU identity；CUDA unavailable 时不要继续 model setup。

```bash
cd "$HOME/5703tts"

third_party/sglang-omni/.venv/bin/python - <<'PY'
import torch

available = torch.cuda.is_available()

print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.is_available()={available}")

if not available:
    raise SystemExit("ERROR: Higgs/SGLang environment cannot access CUDA")

print(f"GPU={torch.cuda.get_device_name(0)}")
print(f"GPU_count={torch.cuda.device_count()}")
PY
```

关键检查：historical Torch CUDA build marker 是 `13.0`；保存实际输出，不能只写预期值。

## 11. Reuse / install HF CLI

若 Part 2 已安装 `hf` 则直接复用，否则以 user-level uv tool 安装；public checkpoint 不要求预先 login。

```bash
export PATH="$HOME/.local/bin:$PATH"

if ! command -v hf >/dev/null 2>&1; then
  uv tool install "huggingface_hub[hf_xet]"
fi

hf version
```

关键检查：`hf version` 成功；仅在 authentication/rate-limit download failure 时运行 `hf auth login`、`hf auth whoami`，不要把 token 写入 Git、YAML、scripts、logs 或 docs examples。

## 12. 下载 frozen Higgs checkpoint

Remote model、revision 和 local directory 都固定；current upstream 的 `bosonai/higgs-audio-v3-tts-4b` 不属于首次 validation。

```bash
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/5703tts"

mkdir -p models/higgs-tts-3-4b

hf download \
  bosonai/higgs-tts-3-4b \
  --revision 0056125158f940389ab0808a581b8b2c590b32d4 \
  --local-dir models/higgs-tts-3-4b

printf '%s\n' \
  'remote_model=bosonai/higgs-tts-3-4b' \
  'revision=0056125158f940389ab0808a581b8b2c590b32d4' \
  'local_directory=models/higgs-tts-3-4b'
```

关键检查：保留 `hf download` 和三行 identity 输出作为 Cloud evidence。

## 13. 验证 Higgs model files

Model 约 9.3 GB，但 arbitrary size 不是 gate；文件存在和下一节 SHA 才是 mandatory checks。

```bash
cd "$HOME/5703tts"

test -f models/higgs-tts-3-4b/config.json
test -f models/higgs-tts-3-4b/model.safetensors
test -f models/higgs-tts-3-4b/model.safetensors.index.json
test -f models/higgs-tts-3-4b/tokenizer.json
test -f models/higgs-tts-3-4b/tokenizer_config.json

du -sh models/higgs-tts-3-4b
echo 'HIGGS_MODEL_FILES_OK=1'
```

关键检查：五个 artifacts 存在，最后输出 `HIGGS_MODEL_FILES_OK=1`。

## 14. 验证 frozen weight SHA-256

对 `model.safetensors` 做 automatic equality gate；不匹配时必须停止，不得对不同 checkpoint 运行 production validation。

```bash
set -euo pipefail

cd "$HOME/5703tts"

MODEL_FILE="models/higgs-tts-3-4b/model.safetensors"
EXPECTED_SHA256="2f7965264c360b38180885006944aa16bd1de20f4e6cff79f6473bfcf8ae3d5a"
ACTUAL_SHA256="$(
  sha256sum "$MODEL_FILE" |
  awk '{print $1}'
)"

printf 'expected=%s\n' "$EXPECTED_SHA256"
printf 'actual=%s\n' "$ACTUAL_SHA256"
test "$ACTUAL_SHA256" = "$EXPECTED_SHA256"

echo 'HIGGS_MODEL_SHA256_OK=1'
```

关键检查：expected 与 actual 完全相同，最后输出 `HIGGS_MODEL_SHA256_OK=1`。

## 15. 创建 local Higgs Cloud config

不要修改 checked-in `config/config.yaml` 或 `config/config.higgs.example.yaml`；backend 会相对 repository root 解析下列 paths，本地 config 不得 commit。

```bash
cd "$HOME/5703tts"

cat > config/config_higgs_cloud.yaml <<'EOF'
tts:
  engine: higgs
  higgs:
    server_executable: third_party/sglang-omni/.venv/bin/sgl-omni
    model_dir: models/higgs-tts-3-4b
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
EOF

sed -n '1,80p' config/config_higgs_cloud.yaml

grep -qxF '/config/config_higgs_cloud.yaml' .git/info/exclude || \
  echo '/config/config_higgs_cloud.yaml' >> .git/info/exclude
```

关键检查：config 固定 `higgs`、`127.0.0.1:18080`、timeouts `900/300`；只改 Cloud checkout 的 `.git/info/exclude`，不改 repository `.gitignore`。

## 16. 验证 config path contract

Current preflight 要求 executable file、local model directory 和 FFmpeg；`model_dir` 不能写 Hugging Face model ID。

```bash
set -euo pipefail

cd "$HOME/5703tts"

test -x third_party/sglang-omni/.venv/bin/sgl-omni
test -d models/higgs-tts-3-4b
test -f models/higgs-tts-3-4b/model.safetensors
test -f config/config_higgs_cloud.yaml

SERVER="$(
  realpath third_party/sglang-omni/.venv/bin/sgl-omni
)"
MODEL_DIR="$(
  realpath models/higgs-tts-3-4b
)"
FFMPEG="$(
  realpath "$(command -v ffmpeg)"
)"

printf 'server_executable=%s\n' "$SERVER"
printf 'model_dir=%s\n' "$MODEL_DIR"
printf 'ffmpeg=%s\n' "$FFMPEG"
```

关键检查：三项都打印 absolute paths；production policy 仍是 Higgs primary、CosyVoice3 explicit backup、无 automatic fallback。

## 17. 检查 TCP port 18080

安装交接前确认 local worker 将使用的 port 未被占用；这里不启动任何 server。

```bash
set -euo pipefail

if ss -ltn | awk '{print $4}' | grep -Eq '(^|:)18080$'; then
  echo 'ERROR: TCP port 18080 is already in use' >&2
  exit 1
fi

echo 'HIGGS_PORT_18080_FREE=1'
```

关键检查：最后输出 `HIGGS_PORT_18080_FREE=1`。

## 18. Higgs primary 最终检查

这是 installation 的最终 mandatory gate，不启动 SGLang server、不发送 request，也不 synthesis。

```bash
set -euo pipefail

cd "$HOME/5703tts"

echo '=== NVIDIA DRIVER ==='
nvidia-smi

echo '=== CUDA TOOLKIT ==='
test -x /usr/local/cuda/bin/nvcc
/usr/local/cuda/bin/nvcc --version

echo '=== UCX ==='
command -v ucx_info
ucx_info -v
ucx_info -d | grep -i cuda

echo '=== SGLANG-OMNI ==='
test -x third_party/sglang-omni/.venv/bin/sgl-omni
third_party/sglang-omni/.venv/bin/sgl-omni --help >/dev/null

echo '=== HIGGS PYTHON / CUDA ==='
third_party/sglang-omni/.venv/bin/python - <<'PY'
import torch

assert torch.cuda.is_available(), "Higgs/SGLang CUDA unavailable"

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

echo '=== HIGGS MODEL ==='
test -f models/higgs-tts-3-4b/model.safetensors

EXPECTED='2f7965264c360b38180885006944aa16bd1de20f4e6cff79f6473bfcf8ae3d5a'
ACTUAL="$(
  sha256sum models/higgs-tts-3-4b/model.safetensors |
  awk '{print $1}'
)"
test "$ACTUAL" = "$EXPECTED"

echo '=== HIGGS CONFIG ==='
test -f config/config_higgs_cloud.yaml

echo 'HIGGS_PRIMARY_ENVIRONMENT_OK=1'
```

关键检查：只有所有 mandatory checks 成功后才输出 `HIGGS_PRIMARY_ENVIRONMENT_OK=1`。

Part 3 到此结束。不要手工常驻 `sgl-omni serve`、请求 `/v1/audio/speech`、render turns、materialize Higgs references 或运行 production synthesis；下一步按 [HIGGS_PRODUCTION_RUNBOOK_CN.md](HIGGS_PRODUCTION_RUNBOOK_CN.md) 验证 `5703tts -> backends/higgs.py -> higgs_worker.py -> sgl-omni serve -> /health -> /v1/audio/speech`，server lifecycle 必须由 worker 管理。

最终 environments 必须保持分离，绝不能 merge：

```text
5703tts/
├── .venv/
│   └── Python 3.11 project environment
│
├── third_party/
│   ├── CosyVoice/
│   │   └── .venv/
│   │       └── Python 3.10 CV3 backup environment
│   │
│   └── sglang-omni/
│       └── .venv/
│           └── Python 3.12 Higgs primary environment
│
└── models/
    ├── Fun-CosyVoice3-0.5B/
    └── higgs-tts-3-4b/
```

License operational note：checkpoint 当前标注为 Boson Higgs TTS 3 research/non-commercial license；deployment/distribution 前阅读 frozen checkpoint 的 [LICENSE](https://huggingface.co/bosonai/higgs-tts-3-4b/blob/0056125158f940389ab0808a581b8b2c590b32d4/LICENSE)，本文不构成 legal advice。
