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
# Build tt-metal (if needed)
#==============================================================================

build_tt_metal() {
    cd /tt-metal

    # Check if build is needed
    if [ -f "/tt-metal/build/lib/libtt_metal.so" ] && [ "${SKIP_TT_METAL_BUILD:-0}" = "1" ]; then
        echo "tt-metal already built and SKIP_TT_METAL_BUILD=1, skipping..."
        return 0
    fi

    # Check if this is a fresh source (no build dir or stale)
    if [ ! -f "/tt-metal/build/lib/libtt_metal.so" ]; then
        echo "Building tt-metal C++ components (first run)..."
        echo "This will take a while but is cached for subsequent runs."

        # Install dependencies if not done
        if [ ! -f "/opt/.tt-deps-installed" ]; then
            echo "Installing tt-metal dependencies..."
            ./install_dependencies.sh --docker
            touch /opt/.tt-deps-installed
        fi

        # Build with parallel jobs
        local jobs="${BUILD_JOBS:-16}"
        echo "Building with $jobs parallel jobs..."
        CMAKE_BUILD_PARALLEL_LEVEL=$jobs MAKEFLAGS="-j$jobs" ./build_metal.sh

        echo "tt-metal build complete."
    else
        echo "tt-metal already built, skipping C++ build."
    fi

    # Always create/update the venv with tt-metal Python package
    if [ ! -f "/opt/venv/.tt-metal-installed" ] || [ "/tt-metal/setup.py" -nt "/opt/venv/.tt-metal-installed" ]; then
        echo "Installing tt-metal Python package..."
        cd /tt-metal
        ./create_venv.sh --env-dir /opt/venv --no-create
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
# Hand off to main entrypoint
#==============================================================================

echo ""
echo "=== Dev setup complete, starting vLLM ==="
echo ""

exec /usr/local/bin/entrypoint.sh "$@"
