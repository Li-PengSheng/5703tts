from __future__ import annotations

import stat
import subprocess
from pathlib import Path

SCRIPT = Path("scripts/setup_cloud_environment.sh")


def _shell_function(name: str) -> str:
    script = SCRIPT.read_text(encoding="utf-8")
    start = script.index(f"{name}() {{\n")
    end = script.index("\n}\n", start) + 3
    return script[start:end]


def _run_shell_function(
    name: str, *args: str, stdin: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    source = f'set -Eeuo pipefail\n{_shell_function(name)}\n{name} "$@"'
    return subprocess.run(
        ["bash", "-c", source, "bash", *args],
        input=stdin,
        capture_output=True,
        check=False,
    )


def test_cloud_setup_script_is_executable_bash() -> None:
    assert SCRIPT.is_file()
    assert SCRIPT.read_text(encoding="utf-8").startswith(
        "#!/usr/bin/env bash\n\nset -Eeuo pipefail\n"
    )
    assert SCRIPT.stat().st_mode & stat.S_IXUSR


def test_cloud_setup_script_freezes_reviewed_runtime_identities() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    expected = {
        "COSYVOICE_REPO_URL": "https://github.com/FunAudioLLM/CosyVoice.git",
        "COSYVOICE_COMMIT": "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc",
        "MATCHA_COMMIT": "dd9105b34bf2be2230f4aa1e4769fb586a3c824e",
        "COSYVOICE_MODEL_REVISION": "29e01c4e8d000f4bcd70751be16fa94bf3d85a18",
        "UCX_COMMIT": "d8e50df6651b9ea5b76f23aee0aefbf053a4137a",
        "SGLANG_OMNI_VERSION": "0.1.3",
        "HIGGS_MODEL_REVISION": "0056125158f940389ab0808a581b8b2c590b32d4",
        "HIGGS_MODEL_SHA256": (
            "2f7965264c360b38180885006944aa16bd1de20f4e6cff79f6473bfcf8ae3d5a"
        ),
    }
    for name, value in expected.items():
        assert f'readonly {name}="{value}"' in script


def test_cosyvoice_cleanliness_allows_only_installer_venv() -> None:
    allowed = b"?? .venv/\0?? .venv/bin/python\0?? .venv/lib/site.py\0"
    assert (
        _run_shell_function("cosyvoice_status_is_clean", stdin=allowed).returncode == 0
    )

    for rejected in (
        b"?? notes.txt\0",
        b"?? custom/file.txt\0",
        b" M cosyvoice/foo.py\0",
        b"M  requirements.txt\0",
    ):
        assert (
            _run_shell_function("cosyvoice_status_is_clean", stdin=rejected).returncode
            != 0
        )


def test_ucx_cleanliness_ignores_untracked_but_rejects_tracked_changes() -> None:
    source = _shell_function("ucx_source_tracked_clean")
    assert "diff --quiet --" in source
    assert "diff --cached --quiet --" in source
    assert "status --porcelain" not in source

    assert _run_shell_function("ucx_diff_statuses_are_clean", "0", "0").returncode == 0
    assert _run_shell_function("ucx_diff_statuses_are_clean", "1", "0").returncode != 0
    assert _run_shell_function("ucx_diff_statuses_are_clean", "0", "1").returncode != 0


def test_ucx_version_match_is_exact() -> None:
    assert _run_shell_function("ucx_version_is_compatible", "1.20.1").returncode == 0
    for version in ("1.20.10", "1.20.11", "1.20.0", "1.21.1", "2.20.1"):
        assert _run_shell_function("ucx_version_is_compatible", version).returncode != 0


def test_cuda_completed_marker_fails_closed_before_installer_rerun() -> None:
    function = _shell_function("ensure_cuda_toolkit")
    marker_check = function.index('[[ -f "$CUDA_COMPLETED_MARKER" ]]')
    installer_call = function.index("install_cuda \\\n")
    marker_write = function.index('>"$CUDA_COMPLETED_MARKER"')

    assert marker_check < installer_call < marker_write
    assert "refusing to rerun install_cuda automatically" in function
