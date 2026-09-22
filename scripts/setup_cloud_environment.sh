#!/usr/bin/env bash

set -Eeuo pipefail

CURRENT_STAGE="startup"
readonly REBOOT_REQUIRED_EXIT=75
readonly COSYVOICE_REPO_URL="https://github.com/FunAudioLLM/CosyVoice.git"
readonly COSYVOICE_COMMIT="074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
readonly MATCHA_COMMIT="dd9105b34bf2be2230f4aa1e4769fb586a3c824e"
readonly COSYVOICE_MODEL_ID="FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
readonly COSYVOICE_MODEL_REVISION="29e01c4e8d000f4bcd70751be16fa94bf3d85a18"
readonly UCX_COMMIT="d8e50df6651b9ea5b76f23aee0aefbf053a4137a"
readonly SGLANG_OMNI_VERSION="0.1.3"
readonly HIGGS_MODEL_ID="bosonai/higgs-tts-3-4b"
readonly HIGGS_MODEL_REVISION="0056125158f940389ab0808a581b8b2c590b32d4"
readonly HIGGS_MODEL_SHA256="2f7965264c360b38180885006944aa16bd1de20f4e6cff79f6473bfcf8ae3d5a"
readonly -a BASE_PACKAGES=(
  git curl wget ca-certificates python3 ffmpeg sox libsndfile1
  build-essential pkg-config iproute2 mokutil pciutils debconf
)
readonly -a UCX_PACKAGES=(
  autoconf automake libtool make gcc g++ flex bison m4 libnuma-dev
  libibverbs-dev librdmacm-dev rdma-core libnl-3-dev libnl-route-3-dev
  libudev-dev
)

ORIGINAL_ARGS=("$@")
MODE="all"
MODE_SET=0
NO_REBOOT=0

on_error() {
  local status="$1"
  local line="$2"
  local command="$3"
  trap - ERR
  printf 'ERROR: stage=%s line=%s exit=%s command=%q\n' \
    "$CURRENT_STAGE" "$line" "$status" "$command" >&2
  exit "$status"
}

trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_cloud_environment.sh [MODE] [--no-reboot]

Modes:
  --all        Install/resume base, CosyVoice backup, and Higgs primary (default)
  --base       Install/resume only the base environment
  --cosyvoice  Verify base prerequisites, then install/resume CosyVoice
  --higgs      Verify base prerequisites, then install/resume Higgs
  --check      Read-only status check; install or change nothing

Options:
  --no-reboot  Report a required reboot and exit 75 instead of rebooting
  -h, --help   Show this help
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

stage() {
  CURRENT_STAGE="$2"
  printf '\n[%s/12] %s\n' "$1" "$2"
}

while (($#)); do
  case "$1" in
    --all|--base|--cosyvoice|--higgs|--check)
      ((MODE_SET == 0)) || die "choose only one mode"
      MODE="${1#--}"
      MODE_SET=1
      ;;
    --no-reboot)
      NO_REBOOT=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown option: $1"
      ;;
  esac
  shift
done

((EUID != 0)) || die "run this script as the normal sudo-capable SSH user, not root"

SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &&
    pwd
)"
PROJECT_ROOT="$(
  cd -- "$SCRIPT_DIR/.." &&
    pwd
)"

[[ -f "$PROJECT_ROOT/pyproject.toml" ]] || die "pyproject.toml not found under $PROJECT_ROOT"
[[ -f "$PROJECT_ROOT/uv.lock" ]] || die "uv.lock not found under $PROJECT_ROOT"
[[ -d "$PROJECT_ROOT/src/tts5703" ]] || die "src/tts5703 not found under $PROJECT_ROOT"
git -C "$PROJECT_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
  die "$PROJECT_ROOT is not a Git checkout"

cd "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1

# Keep package provisioning non-interactive on cloud/minimal Ubuntu.
# sudo does not reliably preserve these variables, so root package/installer
# commands below also pass them explicitly through `sudo env`.
export DEBIAN_FRONTEND=noninteractive
export DEBCONF_NONINTERACTIVE_SEEN=true
export APT_LISTCHANGES_FRONTEND=none
export NEEDRESTART_MODE=a

readonly PROJECT_ROOT SCRIPT_DIR
readonly PROJECT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
readonly COSYVOICE_DIR="$PROJECT_ROOT/third_party/CosyVoice"
readonly COSYVOICE_PYTHON="$COSYVOICE_DIR/.venv/bin/python"
readonly COSYVOICE_MODEL_DIR="$PROJECT_ROOT/models/Fun-CosyVoice3-0.5B"
readonly UCX_SRC="$HOME/src/ucx"
readonly HIGGS_VENV="$PROJECT_ROOT/third_party/sglang-omni/.venv"
readonly HIGGS_PYTHON="$HIGGS_VENV/bin/python"
readonly SGLANG_EXECUTABLE="$HIGGS_VENV/bin/sgl-omni"
readonly HIGGS_MODEL_DIR="$PROJECT_ROOT/models/higgs-tts-3-4b"
readonly HIGGS_CONFIG="$PROJECT_ROOT/config/config_higgs_cloud.yaml"
readonly SETUP_STATE_DIR="$HOME/.cache/5703tts-setup"
readonly CUDA_COMPLETED_MARKER="$SETUP_STATE_DIR/cuda_install_completed"

REPOSITORY_SHA="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
readonly REPOSITORY_SHA
printf 'Repository root: %s\n' "$PROJECT_ROOT"
printf 'Repository SHA: %s\n' "$REPOSITORY_SHA"

print_resume_command() {
  printf 'After reconnect run:\n'
  printf '  cd %q\n' "$PROJECT_ROOT"
  printf '  bash scripts/setup_cloud_environment.sh'
  if ((${#ORIGINAL_ARGS[@]})); then
    printf ' %q' "${ORIGINAL_ARGS[@]}"
  fi
  printf '\n'
}

request_reboot() {
  local reason="$1"
  printf '\n%s\n' "$reason"
  printf 'SSH will disconnect.\n'
  print_resume_command
  if ((NO_REBOOT)); then
    printf 'Automatic reboot disabled; run sudo reboot, then reconnect.\n' >&2
    exit "$REBOOT_REQUIRED_EXIT"
  fi
  sync
  sudo reboot
  exit "$REBOOT_REQUIRED_EXIT"
}

os_supported() (
  [[ -r /etc/os-release ]] || exit 1
  # shellcheck source=/etc/os-release
  . /etc/os-release
  [[ "$ID" == "ubuntu" && ("$VERSION_ID" == "22.04" || "$VERSION_ID" == "24.04") ]]
)

secure_boot_disabled() {
  command -v mokutil >/dev/null 2>&1 &&
    mokutil --sb-state 2>/dev/null | grep -qi '^SecureBoot disabled$'
}

all_commands_exist() {
  local command_name
  for command_name in "$@"; do
    command -v "$command_name" >/dev/null 2>&1 || return 1
  done
}

python_minor_is() {
  local python_bin="$1"
  local expected="$2"
  [[ -x "$python_bin" ]] &&
    [[ "$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "$expected" ]]
}

project_environment_ready() {
  python_minor_is "$PROJECT_PYTHON" "3.11" &&
    [[ -x "$PROJECT_ROOT/.venv/bin/5703tts" ]] &&
    "$PROJECT_ROOT/.venv/bin/5703tts" --help >/dev/null 2>&1
}

base_commands_ready() {
  all_commands_exist git curl wget python3 ffmpeg sox ss mokutil
}

packages_installed() {
  local package
  for package in "$@"; do
    dpkg-query -W -f='${Status}\n' "$package" 2>/dev/null | \
      grep -qx 'install ok installed' || return 1
  done
}

base_environment_ready() {
  os_supported &&
    secure_boot_disabled &&
    base_commands_ready &&
    nvidia-smi >/dev/null 2>&1 &&
    command -v uv >/dev/null 2>&1 &&
    project_environment_ready
}

disk_check() {
  stage 1 "Checking Ubuntu and disk space"
  os_supported || die "supported OS is Ubuntu 22.04 or 24.04 only"
  cat /etc/os-release
  uname -a
  df -h "$PROJECT_ROOT"

  local available_kb
  available_kb="$(df -Pk "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')"
  if ((available_kb < 5 * 1024 * 1024)); then
    die "less than 5 GiB is free; expand the disk before installing models and runtimes"
  elif ((available_kb < 30 * 1024 * 1024)); then
    warn "free space is below 30 GiB; the guide recommends about 100 GiB minimum and 150 GiB safer headroom"
  fi
}

sudo_noninteractive() {
  sudo env \
    DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NONINTERACTIVE_SEEN=true \
    APT_LISTCHANGES_FRONTEND=none \
    NEEDRESTART_MODE=a \
    "$@"
}

configure_keyboard_us() {
  printf 'Configuring non-interactive US keyboard layout.\n'

  # debconf-set-selections is supplied by debconf. Minimal images normally
  # already contain it; bootstrap it explicitly if necessary.
  if ! command -v debconf-set-selections >/dev/null 2>&1; then
    sudo_noninteractive apt-get update
    sudo_noninteractive apt-get install -y debconf
  fi

  sudo debconf-set-selections <<'EOF'
keyboard-configuration keyboard-configuration/modelcode string pc105
keyboard-configuration keyboard-configuration/layoutcode string us
keyboard-configuration keyboard-configuration/variantcode string
keyboard-configuration keyboard-configuration/optionscode string
keyboard-configuration keyboard-configuration/store_defaults_in_debconf_db boolean true
EOF

  sudo mkdir -p /etc/default
  sudo tee /etc/default/keyboard >/dev/null <<'EOF'
XKBMODEL="pc105"
XKBLAYOUT="us"
XKBVARIANT=""
XKBOPTIONS=""
BACKSPACE="guess"
EOF
}

install_base_packages() {
  stage 2 "Installing base packages and checking Secure Boot"
  command -v sudo >/dev/null 2>&1 || die "sudo is required"

  configure_keyboard_us

  if packages_installed "${BASE_PACKAGES[@]}"; then
    printf 'Base packages already installed; skipping apt.\n'
  else
    sudo_noninteractive apt-get update
    sudo_noninteractive apt-get install -y \
      "${BASE_PACKAGES[@]}"
  fi

  base_commands_ready || die "one or more required base commands are unavailable"
  mokutil --sb-state
  secure_boot_disabled || die "Secure Boot must be disabled before NVIDIA installation"
}

ensure_cuda_installer() {
  if [[ ! -f "$HOME/cuda_installer.pyz" ]]; then
    curl -fL \
      https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz \
      -o "$HOME/cuda_installer.pyz"
  fi
}

ensure_nvidia_driver() {
  stage 3 "Checking NVIDIA driver"
  if nvidia-smi; then
    printf 'NVIDIA driver already works; skipping installation.\n'
    return
  fi

  ensure_cuda_installer
  sudo systemctl stop google-cloud-ops-agent 2>/dev/null || true
  printf 'The NVIDIA installer may reboot this VM.\n'
  print_resume_command
  sudo_noninteractive python3 "$HOME/cuda_installer.pyz" install_driver \
    --installation-mode=repo \
    --installation-branch=prod
  request_reboot "NVIDIA driver stage requested reboot."
}

ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi

    local path_line
    path_line='export PATH="$HOME/.local/bin:$PATH"'

    grep -qxF "$path_line" "$HOME/.profile" 2>/dev/null || \
    printf '%s\n' "$path_line" >>"$HOME/.profile"

    grep -qxF "$path_line" "$HOME/.bashrc" 2>/dev/null || \
    printf '%s\n' "$path_line" >>"$HOME/.bashrc"

    uv --version
}

ensure_uv_python() {
  local version="$1"
  if ! uv python find "$version" >/dev/null 2>&1; then
    uv python install "$version"
  fi
  uv python find "$version"
}

ensure_venv() {
  local venv_dir="$1"
  local version="$2"
  if [[ -e "$venv_dir" && ! -x "$venv_dir/bin/python" ]]; then
    die "existing path is not a usable virtual environment: $venv_dir"
  fi
  if [[ ! -x "$venv_dir/bin/python" ]]; then
    uv venv --python "$version" "$venv_dir"
  fi
  python_minor_is "$venv_dir/bin/python" "$version" || \
    die "$venv_dir must use Python $version"
}

ensure_project_environment() {
  stage 4 "Installing uv and the project Python 3.11 environment"
  ensure_uv
  ensure_uv_python "3.11" >/dev/null
  ensure_venv "$PROJECT_ROOT/.venv" "3.11"
  uv sync --frozen
  uv run python --version
  uv run 5703tts --help >/dev/null
}

base_gate() {
  base_environment_ready || die "base environment is incomplete; run this script with --base or --all first"
  nvidia-smi
  uv --version
  "$PROJECT_PYTHON" --version
  ffmpeg -version | sed -n '1p'
  sox --version
  ss --version || ss -h >/dev/null
  printf 'BASE_ENVIRONMENT_OK=1\n'
}

install_base() {
  disk_check
  install_base_packages
  ensure_nvidia_driver
  ensure_project_environment
  base_gate
}

is_cosyvoice_remote() {
  case "$1" in
    https://github.com/QwenAudio/CosyVoice|https://github.com/QwenAudio/CosyVoice.git|\
      git@github.com:QwenAudio/CosyVoice.git|ssh://git@github.com/QwenAudio/CosyVoice.git|\
      https://github.com/FunAudioLLM/CosyVoice|https://github.com/FunAudioLLM/CosyVoice.git|\
      git@github.com:FunAudioLLM/CosyVoice.git|ssh://git@github.com/FunAudioLLM/CosyVoice.git)
      return 0
      ;;
    *) return 1 ;;
  esac
}

cosyvoice_status_is_clean() {
  local entry path
  while IFS= read -r -d '' entry; do
    [[ "${entry:0:2}" == "??" ]] || return 1
    path="${entry:3}"
    [[ "$path" == .venv/* ]] || return 1
  done
}

cosyvoice_source_clean() {
  git -C "$1" status --porcelain=v1 -z --untracked-files=all |
    cosyvoice_status_is_clean
}

ensure_cosyvoice_venv_excluded() {
  local exclude_path
  exclude_path="$(git -C "$COSYVOICE_DIR" rev-parse --git-path info/exclude)"
  if [[ "$exclude_path" != /* ]]; then
    exclude_path="$COSYVOICE_DIR/$exclude_path"
  fi
  grep -qxF '/.venv/' "$exclude_path" || printf '%s\n' '/.venv/' >>"$exclude_path"
}

cosyvoice_source_ready() {
  [[ -d "$COSYVOICE_DIR/.git" ]] || return 1
  is_cosyvoice_remote "$(git -C "$COSYVOICE_DIR" remote get-url origin 2>/dev/null)" || return 1
  cosyvoice_source_clean "$COSYVOICE_DIR" || return 1
  [[ "$(git -C "$COSYVOICE_DIR" rev-parse HEAD 2>/dev/null)" == "$COSYVOICE_COMMIT" ]] || return 1
  ! git -C "$COSYVOICE_DIR" symbolic-ref -q HEAD >/dev/null 2>&1 || return 1
  [[ -e "$COSYVOICE_DIR/third_party/Matcha-TTS/.git" ]] || return 1
  [[ "$(git -C "$COSYVOICE_DIR/third_party/Matcha-TTS" rev-parse HEAD 2>/dev/null)" == "$MATCHA_COMMIT" ]]
}

ensure_cosyvoice_source() {
  stage 5 "Installing pinned CosyVoice source"
  if [[ ! -e "$COSYVOICE_DIR" ]]; then
    mkdir -p "$(dirname -- "$COSYVOICE_DIR")"
    git clone --recursive "$COSYVOICE_REPO_URL" "$COSYVOICE_DIR"
  fi

  [[ -d "$COSYVOICE_DIR/.git" ]] || die "$COSYVOICE_DIR exists but is not a Git checkout"
  local remote
  remote="$(git -C "$COSYVOICE_DIR" remote get-url origin)"
  is_cosyvoice_remote "$remote" || die "unexpected CosyVoice origin: $remote"
  ensure_cosyvoice_venv_excluded
  cosyvoice_source_clean "$COSYVOICE_DIR" || \
    die "CosyVoice checkout has local modifications; preserve or resolve them manually"

  if [[ "$(git -C "$COSYVOICE_DIR" rev-parse HEAD)" != "$COSYVOICE_COMMIT" ]] || \
    git -C "$COSYVOICE_DIR" symbolic-ref -q HEAD >/dev/null 2>&1; then
    git -C "$COSYVOICE_DIR" checkout --detach "$COSYVOICE_COMMIT"
  fi
  git -C "$COSYVOICE_DIR" submodule sync --recursive
  git -C "$COSYVOICE_DIR" submodule update --init --recursive

  cosyvoice_source_ready || die "CosyVoice or Matcha-TTS pin verification failed"
  printf 'CosyVoice=%s\n' "$(git -C "$COSYVOICE_DIR" rev-parse HEAD)"
  printf 'Matcha-TTS=%s\n' \
    "$(git -C "$COSYVOICE_DIR/third_party/Matcha-TTS" rev-parse HEAD)"
}

cosyvoice_gpu_ready() {
  [[ -x "$COSYVOICE_PYTHON" ]] || return 1
  "$COSYVOICE_PYTHON" - <<'PY' >/dev/null 2>&1
import torch

raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
}

ensure_cosyvoice_environment() {
  stage 6 "Installing the CosyVoice Python 3.10 environment"
  ensure_uv_python "3.10" >/dev/null
  ensure_venv "$COSYVOICE_DIR/.venv" "3.10"
  uv pip install \
    --python "$COSYVOICE_PYTHON" \
    -r "$COSYVOICE_DIR/requirements.txt"

  "$COSYVOICE_PYTHON" - <<'PY'
import torch

available = torch.cuda.is_available()
print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.is_available()={available}")
if not available:
    raise SystemExit("ERROR: CosyVoice environment cannot access CUDA")
print(f"GPU={torch.cuda.get_device_name(0)}")
PY
}

ensure_hf_cli() {
  if ! command -v hf >/dev/null 2>&1; then
    uv tool install "huggingface_hub[hf_xet]"
  fi
  hf version
}

cosyvoice_model_files_ready() {
  [[ -f "$COSYVOICE_MODEL_DIR/cosyvoice3.yaml" ]] &&
    [[ -f "$COSYVOICE_MODEL_DIR/llm.pt" ]] &&
    [[ -f "$COSYVOICE_MODEL_DIR/flow.pt" ]] &&
    [[ -f "$COSYVOICE_MODEL_DIR/hift.pt" ]]
}

cosyvoice_model_ready() {
  cosyvoice_model_files_ready &&
    [[ -f "$COSYVOICE_MODEL_DIR/.cache/huggingface/trees/${COSYVOICE_MODEL_REVISION}.json" ]]
}

download_cosyvoice_model() {
  mkdir -p "$COSYVOICE_MODEL_DIR"
  if cosyvoice_model_ready; then
    printf 'Pinned CosyVoice model already exists; skipping download.\n'
    return
  fi
  if ! hf download \
    "$COSYVOICE_MODEL_ID" \
    --revision "$COSYVOICE_MODEL_REVISION" \
    --local-dir "$COSYVOICE_MODEL_DIR"; then
    printf 'Model download failed. If Hugging Face reports authentication or rate limits, run hf auth login and rerun this script.\n' >&2
    return 1
  fi
  cosyvoice_model_files_ready || die "CosyVoice model download is incomplete"
}

cosyvoice_import_ready() {
  [[ -x "$COSYVOICE_PYTHON" ]] || return 1
  "$COSYVOICE_PYTHON" - "$COSYVOICE_DIR" <<'PY' >/dev/null 2>&1
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "third_party" / "Matcha-TTS"))
sys.path.insert(0, str(root))
from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: F401
PY
}

cosyvoice_config_ready() {
  [[ -x "$PROJECT_PYTHON" ]] || return 1
  "$PROJECT_PYTHON" - "$PROJECT_ROOT/config/config_cosyvoice.yaml" <<'PY' >/dev/null 2>&1
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
cosy = config["tts"]["cosyvoice"]
expected = {
    "python_bin": "third_party/CosyVoice/.venv/bin/python",
    "repo_dir": "third_party/CosyVoice",
    "model_dir": "models/Fun-CosyVoice3-0.5B",
    "load_trt": False,
    "load_vllm": False,
    "fp16": True,
}
raise SystemExit(0 if all(cosy.get(key) == value for key, value in expected.items()) else 1)
PY
}

verify_cosyvoice_import_and_config() {
  cosyvoice_import_ready || die "CosyVoice3 import failed with the production sys.path contract"
  cosyvoice_config_ready || die "config/config_cosyvoice.yaml no longer matches the reviewed contract"
  printf 'CosyVoice3 import and config contract OK.\n'
}

cosyvoice_gate() {
  cosyvoice_source_ready || die "CosyVoice source gate failed"
  python_minor_is "$COSYVOICE_PYTHON" "3.10" || die "CosyVoice Python gate failed"
  cosyvoice_gpu_ready || die "CosyVoice CUDA gate failed"
  cosyvoice_model_files_ready || die "CosyVoice model gate failed"
  cosyvoice_import_ready || die "CosyVoice import gate failed"
  cosyvoice_config_ready || die "CosyVoice config gate failed"
  printf 'COSYVOICE_BACKUP_ENVIRONMENT_OK=1\n'
}

install_cosyvoice() {
  ensure_cosyvoice_source
  ensure_cosyvoice_environment
  stage 7 "Installing the pinned CosyVoice model and checking integration"
  ensure_hf_cli
  download_cosyvoice_model
  verify_cosyvoice_import_and_config
  cosyvoice_gate
}

cuda13_ready() {
  [[ -d /usr/local/cuda ]] &&
    [[ -x /usr/local/cuda/bin/nvcc ]] &&
    /usr/local/cuda/bin/nvcc --version 2>/dev/null | grep -Eq 'release 13\.'
}

print_nvcc_state() {
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    /usr/local/cuda/bin/nvcc --version
  else
    printf 'nvcc=ABSENT\n'
  fi
}

ensure_cuda_toolkit() {
  stage 8 "Checking CUDA 13 Toolkit"
  if cuda13_ready; then
    /usr/local/cuda/bin/nvcc --version
    printf 'CUDA 13 Toolkit already works; skipping installation.\n'
    return
  fi

  if [[ -f "$CUDA_COMPLETED_MARKER" ]]; then
    print_nvcc_state
    die "CUDA installer previously returned successfully, but CUDA 13.x is not verified; reboot first if still pending, otherwise inspect the toolkit; refusing to rerun install_cuda automatically"
  fi

  ensure_cuda_installer
  printf 'The CUDA installer may reboot before installation completes. Rerun the same command after every reconnect.\n'
  print_resume_command
  sudo_noninteractive python3 "$HOME/cuda_installer.pyz" install_cuda \
    --installation-mode=repo \
    --installation-branch=prod
  mkdir -p "$SETUP_STATE_DIR"
  printf 'install_cuda returned successfully; final reboot pending\n' >"$CUDA_COMPLETED_MARKER"
  request_reboot "CUDA Toolkit installation stage completed. A reboot is required before continuing."
}

ucx_version_is_compatible() {
  [[ "$1" =~ (^|[^0-9])1\.20\.1([^0-9]|$) ]]
}

ucx_ready() {
  local version_output
  command -v ucx_info >/dev/null 2>&1 &&
    version_output="$(ucx_info -v 2>/dev/null)" &&
    ucx_version_is_compatible "$version_output" &&
    ucx_info -d 2>/dev/null | grep -qi cuda
}

is_ucx_remote() {
  case "$1" in
    https://github.com/openucx/ucx|https://github.com/openucx/ucx.git|\
      git@github.com:openucx/ucx.git|ssh://git@github.com/openucx/ucx.git)
      return 0
      ;;
    *) return 1 ;;
  esac
}

ucx_diff_statuses_are_clean() {
  [[ "$1" == 0 && "$2" == 0 ]]
}

ucx_source_tracked_clean() {
  local worktree_status staged_status
  git -C "$1" diff --quiet -- && worktree_status=0 || worktree_status=$?
  git -C "$1" diff --cached --quiet -- && staged_status=0 || staged_status=$?
  ucx_diff_statuses_are_clean "$worktree_status" "$staged_status"
}

ensure_ucx_source() {
  if [[ ! -e "$UCX_SRC" ]]; then
    mkdir -p "$(dirname -- "$UCX_SRC")"
    git clone --filter=blob:none https://github.com/openucx/ucx.git "$UCX_SRC"
  fi
  [[ -d "$UCX_SRC/.git" ]] || die "$UCX_SRC exists but is not a Git checkout"

  local remote
  remote="$(git -C "$UCX_SRC" remote get-url origin)"
  is_ucx_remote "$remote" || die "unexpected UCX origin: $remote"
  ucx_source_tracked_clean "$UCX_SRC" || \
    die "UCX tracked source has local modifications; preserve or resolve them manually"
  if [[ "$(git -C "$UCX_SRC" rev-parse HEAD)" != "$UCX_COMMIT" ]] || \
    git -C "$UCX_SRC" symbolic-ref -q HEAD >/dev/null 2>&1; then
    git -C "$UCX_SRC" checkout --detach "$UCX_COMMIT" || \
      die "UCX checkout switch failed; inspect possible untracked-file conflicts manually"
  fi
}

ensure_ucx() {
  stage 9 "Checking or building pinned UCX"
  if ucx_ready; then
    ucx_info -v
    printf 'Compatible CUDA-enabled UCX already works; skipping rebuild.\n'
    return
  fi

  if packages_installed "${UCX_PACKAGES[@]}"; then
    printf 'UCX build packages already installed; skipping apt.\n'
  else
    sudo_noninteractive apt-get update
    sudo_noninteractive apt-get install -y \
      "${UCX_PACKAGES[@]}"
  fi

  ensure_ucx_source
  cd "$UCX_SRC"
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
  cd "$PROJECT_ROOT"

  ucx_ready || die "UCX 1.20.1 with CUDA support was not verified after build"
  ucx_info -v
  ucx_info -d | grep -i cuda
}

sglang_versions_ready() {
  [[ -x "$HIGGS_PYTHON" ]] || return 1
  "$HIGGS_PYTHON" - <<'PY' >/dev/null 2>&1
from importlib.metadata import PackageNotFoundError, version

expected = {
    "sglang-omni": "0.1.3",
    "sglang": "0.5.16",
    "torch": "2.11.0",
    "transformers": "5.12.1",
}
try:
    actual = {package: version(package) for package in expected}
except PackageNotFoundError:
    raise SystemExit(1)
raise SystemExit(0 if actual == expected else 1)
PY
}

print_sglang_versions() {
  "$HIGGS_PYTHON" - <<'PY'
from importlib.metadata import version

for package in ("sglang-omni", "sglang", "torch", "transformers"):
    print(f"{package}={version(package)}")
PY
}

higgs_gpu_ready() {
  [[ -x "$HIGGS_PYTHON" ]] || return 1
  "$HIGGS_PYTHON" - <<'PY' >/dev/null 2>&1
import torch

raise SystemExit(
    0
    if torch.cuda.is_available()
    and torch.__version__.split("+")[0] == "2.11.0"
    and torch.version.cuda == "13.0"
    else 1
)
PY
}

ensure_higgs_environment() {
  stage 10 "Installing the pinned SGLang-Omni environment"
  ensure_uv_python "3.12" >/dev/null
  mkdir -p "$(dirname -- "$HIGGS_VENV")"
  ensure_venv "$HIGGS_VENV" "3.12"

  if ! sglang_versions_ready; then
    uv pip install \
      --python "$HIGGS_PYTHON" \
      --prerelease=allow \
      "sglang-omni==$SGLANG_OMNI_VERSION"
  fi
  sglang_versions_ready || die "SGLang frozen dependency identity check failed"
  [[ -x "$SGLANG_EXECUTABLE" ]] || die "sgl-omni executable is missing"
  "$SGLANG_EXECUTABLE" --help >/dev/null
  print_sglang_versions

  "$HIGGS_PYTHON" - <<'PY'
import torch

available = torch.cuda.is_available()
print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.is_available()={available}")
if not available:
    raise SystemExit("ERROR: Higgs/SGLang environment cannot access CUDA")
if torch.__version__.split("+")[0] != "2.11.0" or torch.version.cuda != "13.0":
    raise SystemExit("ERROR: Higgs frozen Torch/CUDA identity mismatch")
print(f"GPU={torch.cuda.get_device_name(0)}")
print(f"GPU_count={torch.cuda.device_count()}")
PY
}

higgs_model_files_ready() {
  [[ -f "$HIGGS_MODEL_DIR/config.json" ]] &&
    [[ -f "$HIGGS_MODEL_DIR/model.safetensors" ]] &&
    [[ -f "$HIGGS_MODEL_DIR/model.safetensors.index.json" ]] &&
    [[ -f "$HIGGS_MODEL_DIR/tokenizer.json" ]] &&
    [[ -f "$HIGGS_MODEL_DIR/tokenizer_config.json" ]]
}

higgs_model_actual_sha() {
  sha256sum "$HIGGS_MODEL_DIR/model.safetensors" | awk '{print $1}'
}

higgs_model_ready() {
  higgs_model_files_ready || return 1
  if [[ -z "${HIGGS_VERIFIED_SHA:-}" ]]; then
    HIGGS_VERIFIED_SHA="$(higgs_model_actual_sha)"
  fi
  [[ "$HIGGS_VERIFIED_SHA" == "$HIGGS_MODEL_SHA256" ]]
}

download_higgs_model() {
  if [[ -f "$HIGGS_MODEL_DIR/model.safetensors" ]]; then
    HIGGS_VERIFIED_SHA="$(higgs_model_actual_sha)"
    if [[ "$HIGGS_VERIFIED_SHA" != "$HIGGS_MODEL_SHA256" ]]; then
      die "existing Higgs model.safetensors SHA-256 mismatches; refusing to delete or replace it"
    fi
  fi
  if higgs_model_ready; then
    printf 'Pinned Higgs model already exists; skipping download.\n'
    return
  fi

  mkdir -p "$HIGGS_MODEL_DIR"
  if ! hf download \
    "$HIGGS_MODEL_ID" \
    --revision "$HIGGS_MODEL_REVISION" \
    --local-dir "$HIGGS_MODEL_DIR"; then
    printf 'Model download failed. If Hugging Face reports authentication or rate limits, run hf auth login and rerun this script.\n' >&2
    return 1
  fi
  higgs_model_files_ready || die "Higgs model download is incomplete"
  HIGGS_VERIFIED_SHA="$(higgs_model_actual_sha)"
  [[ "$HIGGS_VERIFIED_SHA" == "$HIGGS_MODEL_SHA256" ]] || \
    die "Higgs model SHA-256 mismatch: $HIGGS_VERIFIED_SHA"
}

higgs_config_ready() {
  [[ -x "$PROJECT_PYTHON" && -f "$HIGGS_CONFIG" ]] || return 1
  "$PROJECT_PYTHON" - "$HIGGS_CONFIG" <<'PY' >/dev/null 2>&1
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "tts": {
        "engine": "higgs",
        "higgs": {
            "server_executable": "third_party/sglang-omni/.venv/bin/sgl-omni",
            "model_dir": "models/higgs-tts-3-4b",
            "host": "127.0.0.1",
            "port": 18080,
            "startup_timeout_seconds": 900,
            "inference_timeout_seconds": 300,
            "ffmpeg_bin": "ffmpeg",
        },
    },
    "fade_ms": 5,
    "telephone": {
        "sample_rate": 8000,
        "channels": 1,
        "high_pass_hz": 300,
        "low_pass_hz": 3400,
        "volume_db_reduction": 3,
    },
}
raise SystemExit(0 if config == expected else 1)
PY
}

write_higgs_config() {
  local temp_config
  temp_config="$(mktemp "$PROJECT_ROOT/config/.config_higgs_cloud.yaml.XXXXXX")"
  chmod 600 "$temp_config"
  cat >"$temp_config" <<'EOF'
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

  if [[ -f "$HIGGS_CONFIG" ]] && cmp -s "$temp_config" "$HIGGS_CONFIG"; then
    rm -f "$temp_config"
  elif [[ -f "$HIGGS_CONFIG" ]]; then
    rm -f "$temp_config"
    die "$HIGGS_CONFIG differs from the reviewed local config; refusing to overwrite it"
  else
    mv "$temp_config" "$HIGGS_CONFIG"
    printf 'Wrote %s\n' "$HIGGS_CONFIG"
  fi

  local exclude_path
  exclude_path="$(git -C "$PROJECT_ROOT" rev-parse --git-path info/exclude)"
  if [[ "$exclude_path" != /* ]]; then
    exclude_path="$PROJECT_ROOT/$exclude_path"
  fi
  grep -qxF '/config/config_higgs_cloud.yaml' "$exclude_path" || \
    printf '%s\n' '/config/config_higgs_cloud.yaml' >>"$exclude_path"
  higgs_config_ready || die "local Higgs config verification failed"
}

port_18080_free() {
  local listeners
  if ! listeners="$(ss -ltn 2>/dev/null)"; then
    return 2
  fi
  if awk '{print $4}' <<<"$listeners" | grep -Eq '(^|:)18080$'; then
    return 1
  fi
  return 0
}

higgs_gate() {
  nvidia-smi >/dev/null
  cuda13_ready || die "CUDA 13 Toolkit gate failed"
  ucx_ready || die "UCX gate failed"
  sglang_versions_ready || die "SGLang package gate failed"
  [[ -x "$SGLANG_EXECUTABLE" ]] || die "SGLang executable gate failed"
  "$SGLANG_EXECUTABLE" --help >/dev/null
  higgs_gpu_ready || die "Higgs Torch/CUDA gate failed"
  higgs_model_ready || die "Higgs model SHA gate failed"
  higgs_config_ready || die "Higgs config gate failed"
  port_18080_free || die "TCP port 18080 is already in use; no process was stopped"
  printf 'HIGGS_PRIMARY_ENVIRONMENT_OK=1\n'
}

install_higgs() {
  ensure_cuda_toolkit
  cuda13_ready || die "CUDA Toolkit is not the required 13.x version"
  ensure_ucx
  ensure_higgs_environment
  stage 11 "Installing the pinned Higgs model and local config"
  ensure_hf_cli
  download_higgs_model
  write_higgs_config
  stage 12 "Running the Higgs installation gate"
  higgs_gate
}

report_check() {
  local label="$1"
  local status="$2"
  printf '%-28s %s\n' "$label" "$status"
  [[ "$status" == "PASS" ]] || CHECK_FAILURES=$((CHECK_FAILURES + 1))
}

run_check_mode() {
  CURRENT_STAGE="read-only environment check"
  CHECK_FAILURES=0
  printf '\nRead-only environment status:\n'

  local available_kb
  available_kb="$(df -Pk "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')"
  if ((available_kb < 5 * 1024 * 1024)); then report_check "Disk space" "LOW_SPACE"; else report_check "Disk space" "PASS"; fi

  if os_supported; then report_check "OS" "PASS"; else report_check "OS" "WRONG_VERSION"; fi
  if command -v mokutil >/dev/null 2>&1; then
    if secure_boot_disabled; then report_check "Secure Boot" "PASS"; else report_check "Secure Boot" "WRONG_VERSION"; fi
  else
    report_check "Secure Boot" "MISSING"
  fi
  if base_commands_ready; then report_check "Base commands" "PASS"; else report_check "Base commands" "MISSING"; fi
  if nvidia-smi >/dev/null 2>&1; then report_check "NVIDIA driver" "PASS"; else report_check "NVIDIA driver" "MISSING"; fi
  if command -v uv >/dev/null 2>&1; then report_check "uv" "PASS"; else report_check "uv" "MISSING"; fi
  if project_environment_ready; then report_check "Project Python/env" "PASS"; else report_check "Project Python/env" "NOT_CONFIGURED"; fi

  if cosyvoice_source_ready; then report_check "CosyVoice source" "PASS"; else report_check "CosyVoice source" "NOT_CONFIGURED"; fi
  if python_minor_is "$COSYVOICE_PYTHON" "3.10"; then report_check "CosyVoice Python" "PASS"; else report_check "CosyVoice Python" "NOT_CONFIGURED"; fi
  if cosyvoice_model_ready; then report_check "CosyVoice model" "PASS"; else report_check "CosyVoice model" "MISSING"; fi
  if cosyvoice_gpu_ready; then report_check "CosyVoice GPU" "PASS"; else report_check "CosyVoice GPU" "NOT_CONFIGURED"; fi
  if cosyvoice_import_ready && cosyvoice_config_ready; then
    report_check "CosyVoice import/config" "PASS"
  else
    report_check "CosyVoice import/config" "NOT_CONFIGURED"
  fi

  if cuda13_ready; then
    report_check "CUDA Toolkit" "PASS"
  elif [[ -f "$CUDA_COMPLETED_MARKER" ]]; then
    report_check "CUDA Toolkit" "REBOOT_PENDING_OR_INSTALL_COMPLETED"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    report_check "CUDA Toolkit" "WRONG_VERSION"
  else
    report_check "CUDA Toolkit" "MISSING"
  fi
  if ucx_ready; then report_check "UCX" "PASS"; else report_check "UCX" "NOT_CONFIGURED"; fi
  if sglang_versions_ready && [[ -x "$SGLANG_EXECUTABLE" ]]; then
    report_check "SGLang environment" "PASS"
  else
    report_check "SGLang environment" "NOT_CONFIGURED"
  fi
  if higgs_gpu_ready; then report_check "Higgs GPU" "PASS"; else report_check "Higgs GPU" "NOT_CONFIGURED"; fi
  if higgs_model_ready; then report_check "Higgs model/SHA" "PASS"; else report_check "Higgs model/SHA" "MISSING"; fi
  if higgs_config_ready; then report_check "Local Higgs config" "PASS"; else report_check "Local Higgs config" "NOT_CONFIGURED"; fi
  if command -v ss >/dev/null 2>&1 && port_18080_free; then
    report_check "TCP port 18080" "PASS"
  else
    report_check "TCP port 18080" "NOT_CONFIGURED"
  fi

  if ((CHECK_FAILURES == 0)); then
    printf 'CHECK_COMPLETE=1\n'
    return 0
  fi
  printf 'CHECK_INCOMPLETE=%s\n' "$CHECK_FAILURES" >&2
  exit 1
}

print_summary() {
  local gpu_name cuda_line ucx_line
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
  cuda_line="$(/usr/local/cuda/bin/nvcc --version | grep 'release' | sed -n '1p')"
  ucx_line="$(ucx_info -v | sed -n '1p')"
  printf '\nCloud environment summary:\n'
  printf 'Repository SHA: %s\n' "$REPOSITORY_SHA"
  printf 'Project Python: %s\n' "$("$PROJECT_PYTHON" --version 2>&1)"
  printf 'CosyVoice Python: %s\n' "$("$COSYVOICE_PYTHON" --version 2>&1)"
  printf 'CosyVoice commit: %s\n' "$(git -C "$COSYVOICE_DIR" rev-parse HEAD)"
  printf 'CosyVoice model revision: %s\n' "$COSYVOICE_MODEL_REVISION"
  printf 'Higgs Python: %s\n' "$("$HIGGS_PYTHON" --version 2>&1)"
  printf 'SGLang-Omni: %s\n' "$SGLANG_OMNI_VERSION"
  printf 'Higgs model revision: %s\n' "$HIGGS_MODEL_REVISION"
  if [[ -z "${HIGGS_VERIFIED_SHA:-}" ]]; then
    higgs_model_ready || die "Higgs model SHA summary check failed"
  fi
  printf 'Higgs model SHA: %s\n' "$HIGGS_VERIFIED_SHA"
  printf 'GPU: %s\n' "$gpu_name"
  printf 'CUDA Toolkit: %s\n' "$cuda_line"
  printf 'UCX: %s\n' "$ucx_line"
  printf 'CLOUD_ENVIRONMENT_SETUP_OK=1\n'
  printf 'NEXT:\n%s\n' "$PROJECT_ROOT/docs/HIGGS_PRODUCTION_RUNBOOK_CN.md"
}

case "$MODE" in
  all)
    install_base
    install_cosyvoice
    install_higgs
    print_summary
    ;;
  base)
    install_base
    ;;
  cosyvoice)
    disk_check
    base_gate
    install_cosyvoice
    ;;
  higgs)
    disk_check
    base_gate
    install_higgs
    ;;
  check)
    df -h "$PROJECT_ROOT"
    run_check_mode
    ;;
esac
