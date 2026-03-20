#!/bin/bash
# Dev entrypoint - builds tt-metal and installs vllm from bind-mounted sources
set -eo pipefail

echo "=== Dev Environment Startup ==="

#==============================================================================
# Validate source mounts
#==============================================================================

validate_sources() {
    local missing=()

    if [ ! -f "/tt-metal/build_metal.sh" ]; then
        missing+=("/tt-metal (tt-metal source not mounted)")
    fi

    if [ ! -f "/vllm/setup.py" ] && [ ! -f "/vllm/pyproject.toml" ]; then
        missing+=("/vllm (vllm source not mounted)")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        echo "ERROR: Required source directories not properly mounted:"
        for m in "${missing[@]}"; do
            echo "  - $m"
        done
        echo ""
        echo "Make sure WORKSPACE_PATH is set correctly in dev/.env"
        echo "Expected: WORKSPACE_PATH=/path/to/ws/<workspace-name>"
        exit 1
    fi

    echo "Source directories validated:"
    echo "  tt-metal: /tt-metal"
    echo "  vllm: /vllm"
}

validate_sources

#==============================================================================
# Git safety (bind-mounted worktrees)
#==============================================================================
#
# tt-metal/vllm are typically bind-mounted from the host and owned by a
# different user than the container's default user (root). Newer git versions
# refuse to operate on such repos unless they are marked as safe.
if command -v git &>/dev/null; then
    git config --global --add safe.directory /tt-metal 2>/dev/null || true
    git config --global --add safe.directory /vllm 2>/dev/null || true
fi

#==============================================================================
# Build tt-metal (if needed)
#==============================================================================

build_tt_metal() {
    cd /tt-metal

    # Persist the deps-installed marker in the venv volume so it survives
    # container recreates (e.g. when switching HF_MODEL).
    local deps_marker="/opt/venv/.tt-deps-installed"
    local build_commit_marker="/opt/venv/.tt-metal-built-commit"
    local lib_path="/tt-metal/build_Release/lib/libtt_metal.so"

    # Recent tt-metal requires CMake >= 3.24. The dev container bind-mounts a
    # persistent venv at /opt/venv, so we must ensure the correct CMake is
    # installed into the volume, not just baked into the image.
    source /opt/venv/bin/activate
    local required_cmake="3.24.0"
    local cmake_ver=""
    if command -v cmake &>/dev/null; then
        cmake_ver="$(cmake --version 2>/dev/null | head -n 1 | awk '{print $3}' || true)"
    fi
    local need_ninja=0
    if ! command -v ninja &>/dev/null; then
        need_ninja=1
    fi
    if [ -z "$cmake_ver" ] || [ "$(printf '%s\n' "$required_cmake" "$cmake_ver" | sort -V | head -n 1)" != "$required_cmake" ] || [ "$need_ninja" = "1" ]; then
        echo "Installing/Upgrading CMake + Ninja into /opt/venv (need >= $required_cmake, have ${cmake_ver:-<none>})..."
        python -m pip install "cmake>=$required_cmake" ninja
        hash -r || true
        echo "Using CMake: $(command -v cmake)"
        cmake --version | head -n 1 || true
    fi

    local current_commit=""
    if command -v git &>/dev/null && git rev-parse --is-inside-work-tree &>/dev/null; then
        current_commit="$(git rev-parse HEAD 2>/dev/null || true)"
    fi

    local built_commit=""
    if [ -f "$build_commit_marker" ]; then
        built_commit="$(cat "$build_commit_marker" 2>/dev/null || true)"
    fi

    local needs_rebuild=0
    if [ ! -f "$lib_path" ]; then
        needs_rebuild=1
    elif [ -n "$current_commit" ] && [ "$current_commit" != "$built_commit" ]; then
        needs_rebuild=1
    fi

    # Skip rebuild only when explicitly requested *and* the build matches the
    # current source tree. This avoids stale C++ binaries when switching
    # tt-metal commits in a bind-mounted dev workspace.
    if [ "$needs_rebuild" = "0" ] && [ "${SKIP_TT_METAL_BUILD:-0}" = "1" ]; then
        echo "tt-metal already built for this commit and SKIP_TT_METAL_BUILD=1, skipping..."
        return 0
    fi

    if [ "$needs_rebuild" = "1" ]; then
        if [ -f "$lib_path" ]; then
            echo "Rebuilding tt-metal C++ components (source commit changed)..."
            echo "old_commit=${built_commit:-<unknown>} new_commit=${current_commit:-<unknown>}"
        else
            echo "Building tt-metal C++ components (first run)..."
        fi
        echo "This may take a while but is cached for subsequent runs."

        # Install dependencies if not done
        if [ ! -f "$deps_marker" ]; then
            echo "Installing tt-metal dependencies..."
            ./install_dependencies.sh --docker
            touch "$deps_marker"
        fi

        # Build with parallel jobs
        local jobs="${BUILD_JOBS:-16}"
        echo "Building with $jobs parallel jobs..."
        CMAKE_BUILD_PARALLEL_LEVEL=$jobs MAKEFLAGS="-j$jobs" ./build_metal.sh

        echo "tt-metal build complete."

        if [ -n "$current_commit" ]; then
            echo "$current_commit" >"$build_commit_marker" || true
        fi
    else
        echo "tt-metal already built, skipping C++ build."
    fi

    # Always create/update the venv with tt-metal Python package.
    #
    # We *do not* call tt-metal's create_venv.sh here: this dev image already
    # contains /opt/venv, and create_venv.sh is designed to create a new venv
    # (and will prompt/overwrite). For dev we just want an editable install.
    if [ ! -f "/opt/venv/.tt-metal-installed" ] || \
       [ "/tt-metal/setup.py" -nt "/opt/venv/.tt-metal-installed" ] || \
       [ -f "/tt-metal/pyproject.toml" ] && [ "/tt-metal/pyproject.toml" -nt "/opt/venv/.tt-metal-installed" ] || \
       [ -f "/tt-metal/tt_metal/python_env/requirements-dev.txt" ] && [ "/tt-metal/tt_metal/python_env/requirements-dev.txt" -nt "/opt/venv/.tt-metal-installed" ]; then
        echo "Installing tt-metal Python deps + package (editable)..."
        source /opt/venv/bin/activate

        # This dev container uses a *shared* venv at /opt/venv for both tt-metal
        # and vLLM. tt-metal's requirements-dev.txt is intended for full
        # tt-metal development (docs/sweeps/etc) and includes pins that conflict
        # with vLLM (e.g. pydantic==2.9.2, transformers==4.53.0).
        #
        # For serving, we want:
        # - tt-metal runtime deps (from pyproject.toml via `pip install -e .`)
        # - tt-smi CLI for device reset
        #
        # If you need the full tt-metal dev dependency set, opt in explicitly:
        #   TT_METAL_INSTALL_DEV_REQUIREMENTS=1
        PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"
        if [ "${TT_METAL_INSTALL_DEV_REQUIREMENTS:-0}" = "1" ] && [ -f "/tt-metal/tt_metal/python_env/requirements-dev.txt" ]; then
            uv pip install --extra-index-url "$PYTORCH_INDEX" \
                --index-strategy unsafe-best-match \
                -r /tt-metal/tt_metal/python_env/requirements-dev.txt
        else
            if ! command -v tt-smi &>/dev/null; then
                uv pip install --index-strategy unsafe-best-match "tt-smi==3.0.38"
            fi
        fi
        (cd /tt-metal && uv pip install -e . --index-strategy unsafe-best-match)
        touch /opt/venv/.tt-metal-installed
    fi
}

build_tt_metal

#==============================================================================
# Install vllm in editable mode
#==============================================================================

install_vllm() {
    cd /vllm

    # Check if vllm needs reinstall (setup.py or pyproject.toml changed)
    local needs_install=0

    if [ ! -f "/opt/venv/.vllm-installed" ]; then
        needs_install=1
    elif [ -f "/vllm/setup.py" ] && [ "/vllm/setup.py" -nt "/opt/venv/.vllm-installed" ]; then
        needs_install=1
    elif [ -f "/vllm/pyproject.toml" ] && [ "/vllm/pyproject.toml" -nt "/opt/venv/.vllm-installed" ]; then
        needs_install=1
    fi

    # Guardrail: vLLM requires pydantic>=2.10. The tt-metal dev requirements
    # pin pydantic==2.9.2 (explicitly incompatible with vLLM on Python 3.10).
    # If the shared venv got contaminated, force a reinstall so vLLM can bring
    # deps back to a compatible set.
    if [ "$needs_install" = "0" ]; then
        if ! python - <<'PY' >/dev/null 2>&1
import pydantic
parts = pydantic.__version__.split(".")
major = int(parts[0]) if parts else 0
minor = int(parts[1]) if len(parts) > 1 else 0
raise SystemExit(0 if (major, minor) >= (2, 10) else 1)
PY
        then
            needs_install=1
        fi
    fi

    if [ "$needs_install" = "1" ]; then
        echo "Installing vllm in editable mode..."
        source /opt/venv/bin/activate

        export VLLM_TARGET_DEVICE=tt
        uv pip install -e . \
            --extra-index-url https://download.pytorch.org/whl/cpu \
            --index-strategy unsafe-best-match

        touch /opt/venv/.vllm-installed
        echo "vllm installed in editable mode."
    else
        echo "vllm already installed, skipping..."
        echo "(Python changes are picked up automatically in editable mode)"
    fi
}

install_vllm

#==============================================================================
# Hand off to main entrypoint (or exec a custom command)
#==============================================================================

if [ "${DEV_RUN_CMD:-0}" = "1" ]; then
    if [ $# -lt 1 ]; then
        echo "ERROR: DEV_RUN_CMD=1 requires a command to exec (e.g. 'bash -lc ...')." >&2
        exit 2
    fi
    echo ""
    echo "=== Dev setup complete, exec'ing custom command ==="
    echo "cmd: $@"
    echo ""
    exec "$@"
fi

# Speculative decoding config (MTP draft tokens)
if [ -n "${VLLM_SPECULATIVE_CONFIG:-}" ]; then
    set -- "--speculative-config" "${VLLM_SPECULATIVE_CONFIG}" "$@"
    echo "Speculative decoding enabled: ${VLLM_SPECULATIVE_CONFIG}"
fi

echo ""
echo "=== Dev setup complete, starting vLLM ==="
echo ""

# Patch: disable GLM-4.7 thinking for ALL GLM-4.7 variants (not just Flash).
# The image entrypoint only matches GLM-4.7-Flash. This catches GLM-4.7 Full too.
if [ "${GLM_DEFAULT_DISABLE_THINKING:-0}" = "1" ] \
    && echo "${HF_MODEL:-}" | grep -qE "^zai-org/GLM-4\.7" \
    && ! echo " $* " | grep -q -- " --chat-template "; then
    export GLM_CHAT_TEMPLATE_FILE="${GLM_CHAT_TEMPLATE_FILE:-/tmp/glm47_default_no_think.jinja}"
    if python3 - <<'PY' >/dev/null 2>&1
import os
from transformers import AutoTokenizer
model_id = os.environ.get("HF_MODEL", "")
output_path = os.environ.get("GLM_CHAT_TEMPLATE_FILE", "/tmp/glm47_default_no_think.jinja")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
chat_template = getattr(tokenizer, "chat_template", None)
if not chat_template:
    raise RuntimeError("No chat_template")
# Replace the generation prompt to output empty string instead of </think>
# when thinking is disabled. The old patch produced </think> which confuses
# the model (closing tag with no opening tag, especially with tool prompts).
old = "{{- '</think>' if (enable_thinking is defined and not enable_thinking) else '<think>' -}}"
new = "{{- '' if (enable_thinking is not defined or not enable_thinking) else '<think>' -}}"
patched = chat_template.replace(old, new)
with open(output_path, "w") as f:
    f.write(patched)
PY
    then
        echo "Using GLM chat template with thinking disabled: ${GLM_CHAT_TEMPLATE_FILE}"
        set -- "--chat-template" "${GLM_CHAT_TEMPLATE_FILE}" "$@"
    else
        echo "WARNING: Could not generate GLM no-thinking template"
    fi
fi

exec /usr/local/bin/entrypoint.sh "$@"
