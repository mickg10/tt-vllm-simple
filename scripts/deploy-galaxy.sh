#!/usr/bin/env bash
# deploy-galaxy.sh — Deploy or diagnose GLM-4.7-Flash on a Galaxy Wormhole machine
#
# Modes:
#   run      (default) Clone repos, build, start vLLM, wait for healthy, verify
#   diagnostics Clone repos, reset devices, run tt-triage (no vLLM)
#   shell       Clone repos, drop into interactive shell with tt-tools (no vLLM)
#
# Usage:
#   ./deploy-galaxy.sh                           # run mode (starts vLLM)
#   ./deploy-galaxy.sh --mode diagnostics        # diagnostics only (no vLLM)
#   ./deploy-galaxy.sh --mode shell              # interactive shell with tt-tools
#   ./deploy-galaxy.sh --deploy-dir /path/to/ws  # custom workspace
#
# Prerequisites:
#   - Docker access (docker compose must work)
#   - /dev/tenstorrent (TT devices)
#   - /dev/hugepages-1G mounted (run mode)
#   - ~/.cache/huggingface with zai-org/GLM-4.7-Flash or HF_TOKEN set (run mode)
#
# Idempotent: re-running skips already-completed steps.
set -euo pipefail

#=============================================================================
# Configuration (override with environment variables or CLI flags)
#=============================================================================
MODE="${MODE:-run}"
DEPLOY_DIR="${DEPLOY_DIR:-${HOME}/src_docker/ws/glm47_flash_deploy}"
BRANCH="${BRANCH:-galaxy_wormhole}"
TT_METAL_REPO="${TT_METAL_REPO:-https://github.com/mickg10/tt-metal.git}"
VLLM_REPO="${VLLM_REPO:-https://github.com/mickg10/vllm.git}"
DOCKER_TT_REPO="${DOCKER_TT_REPO:-https://github.com/mickg10/tt-vllm-simple.git}"
DEVICE_PROJECT="${DEVICE_PROJECT:-glm-flash}"
BUILD_JOBS="${BUILD_JOBS:-32}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --deploy-dir) DEPLOY_DIR="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --project-name) DEVICE_PROJECT="$2"; shift 2 ;;
        --build-jobs) BUILD_JOBS="$2"; shift 2 ;;
        -h|--help)
            cat <<'USAGE'
Usage: deploy-galaxy.sh [OPTIONS]

Modes:
  --mode run           (default) Build + start vLLM, wait for healthy, verify
  --mode diagnostics   Run tt-triage diagnostics only (no vLLM)
  --mode shell         Interactive shell with tt-tools (no vLLM)

Options:
  --deploy-dir DIR     Workspace directory (default: ~/src_docker/ws/glm47_flash_deploy)
  --branch BRANCH      Git branch (default: galaxy_wormhole)
  --project-name NAME  Docker compose project name (default: glm-flash)
  --build-jobs N       Parallel build jobs (default: 32)
  -h, --help           Show this help

Examples:
  # Fresh deploy — clone repos, build C++, start vLLM:
  ./deploy-galaxy.sh

  # Diagnostics only — clone repos, run check_eth_status + check_noc_status:
  ./deploy-galaxy.sh --mode diagnostics

  # Interactive shell with all tt-tools:
  ./deploy-galaxy.sh --mode shell

  # Deploy to custom directory:
  ./deploy-galaxy.sh --deploy-dir /home/user/my_workspace
USAGE
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

case "$MODE" in
    run|diagnostics|shell) ;;
    *) echo "FATAL: Unknown mode '$MODE'. Use 'run', 'diagnostics', or 'shell'."; exit 1 ;;
esac

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "FATAL: $*" >&2; exit 1; }

#=============================================================================
# Step 0: Validate environment
#=============================================================================
log "=== GLM-4.7-Flash Galaxy — mode: $MODE ==="
log "Workspace: $DEPLOY_DIR"
log "Branch: $BRANCH"

command -v git >/dev/null 2>&1 || die "git not found"
docker compose version >/dev/null 2>&1 || die "'docker compose' not available"

if [ "$MODE" = "run" ] && [ ! -d /dev/hugepages-1G ]; then
    log "WARNING: /dev/hugepages-1G not found — container may fail to start"
fi

#=============================================================================
# Step 1: Clone repositories
#=============================================================================
mkdir -p "$DEPLOY_DIR"

clone_repo() {
    local repo_url="$1" target_dir="$2" branch="$3"
    if [ -d "$target_dir/.git" ]; then
        log "  $(basename "$target_dir") exists, fetching..."
        (
            cd "$target_dir"
            git fetch origin "$branch" --depth=1 2>/dev/null || git fetch origin "$branch"
            git checkout "$branch" 2>/dev/null || git checkout -b "$branch" "origin/$branch"
            git reset --hard "origin/$branch"
        )
    else
        log "  Cloning $(basename "$target_dir")..."
        git clone --depth=1 --branch "$branch" "$repo_url" "$target_dir"
    fi
}

log "Step 1: Cloning repositories..."
clone_repo "$TT_METAL_REPO" "$DEPLOY_DIR/tt-metal" "$BRANCH"
clone_repo "$VLLM_REPO" "$DEPLOY_DIR/vllm" "$BRANCH"
clone_repo "$DOCKER_TT_REPO" "$DEPLOY_DIR/docker_tt" "$BRANCH"

# Submodules — clean stale root-owned tracy dir if Docker created it
TRACY_DIR="$DEPLOY_DIR/tt-metal/tt_metal/third_party/tracy"
if [ -d "$TRACY_DIR" ] && [ ! -f "$TRACY_DIR/CMakeLists.txt" ]; then
    log "  Cleaning stale tracy dir..."
    rm -rf "$TRACY_DIR" 2>/dev/null || \
        docker run --rm -v "$(dirname "$TRACY_DIR"):/mnt" alpine rm -rf /mnt/tracy 2>/dev/null || \
        log "  WARNING: Could not remove stale tracy dir (may need manual cleanup)"
fi
log "  Initializing submodules..."
cd "$DEPLOY_DIR/tt-metal"
git submodule update --init --depth=1 \
    tt_metal/third_party/tt_llk \
    tt_metal/third_party/umd \
    tt_metal/third_party/tracy 2>/dev/null || \
    git submodule update --init \
        tt_metal/third_party/tt_llk \
        tt_metal/third_party/umd \
        tt_metal/third_party/tracy

#=============================================================================
# Step 2: Configure env
#=============================================================================
log "Step 2: Configuring environment..."
cd "$DEPLOY_DIR/docker_tt"

ENV_FILE="dev/.env.glm47.galaxy"
if [ ! -f "$ENV_FILE" ]; then
    die "Env file not found: $ENV_FILE — is docker_tt clone complete?"
fi
sed -i "s|^WORKSPACE_PATH=.*|WORKSPACE_PATH=$DEPLOY_DIR|" "$ENV_FILE"

COMPOSE_BASE="docker compose --env-file $ENV_FILE -f dev/docker-compose.yml -f dev/docker-compose.galaxy.yml"

#=============================================================================
# Mode: diagnostics
#=============================================================================
if [ "$MODE" = "diagnostics" ]; then
    log ""
    log "Step 3: Stopping any running vllm-tt containers..."
    $COMPOSE_BASE down 2>/dev/null || true
    docker ps -q --filter "name=vllm-tt" | xargs -r docker stop 2>/dev/null || true
    rm -f /dev/shm/TT_UMD_LOCK.* 2>/dev/null || true

    log ""
    log "Step 4: Resetting devices..."
    tt-smi -glx_reset 2>/dev/null || tt-smi -r 2>/dev/null || \
        log "WARNING: Device reset not available"
    rm -f /dev/shm/TT_UMD_LOCK.* 2>/dev/null || true

    log ""
    log "Step 5: Running diagnostics (foreground)..."
    $COMPOSE_BASE -f dev/docker-compose.diagnostics.yml run --rm vllm-tt

    log ""
    log "Step 6: Resetting devices (clean state)..."
    rm -f /dev/shm/TT_UMD_LOCK.* 2>/dev/null || true
    tt-smi -glx_reset 2>/dev/null || tt-smi -r 2>/dev/null || true
    rm -f /dev/shm/TT_UMD_LOCK.* 2>/dev/null || true

    log ""
    log "=== Diagnostics complete. Devices reset. ==="
    log "To start vLLM: cd $DEPLOY_DIR/docker_tt && make run-device"
    exit 0
fi

#=============================================================================
# Mode: shell
#=============================================================================
if [ "$MODE" = "shell" ]; then
    log ""
    log "Step 3: Starting interactive shell..."
    $COMPOSE_BASE -f dev/docker-compose.shell.yml run --rm vllm-tt

    log ""
    log "=== Shell session ended. ==="
    log "To start vLLM: cd $DEPLOY_DIR/docker_tt && make run-device"
    exit 0
fi

#=============================================================================
# Mode: run
#=============================================================================
sed -i 's/^SKIP_TT_METAL_BUILD=1/SKIP_TT_METAL_BUILD=0/' "$ENV_FILE"
sed -i "s/^BUILD_JOBS=.*/BUILD_JOBS=$BUILD_JOBS/" "$ENV_FILE"

log "  WORKSPACE_PATH=$DEPLOY_DIR"
log "  SKIP_TT_METAL_BUILD=0 (first run builds C++)"

MAKE_ARGS="DEVICE_PROJECT=$DEVICE_PROJECT"

log "Step 3: Building + starting..."
make run-device $MAKE_ARGS

log "Step 4: Waiting for healthy..."
make wait-device $MAKE_ARGS

# Future restarts skip C++ build
sed -i 's/^SKIP_TT_METAL_BUILD=0/SKIP_TT_METAL_BUILD=1/' "$ENV_FILE"

log "Step 5: Verifying output..."
make verify-device $MAKE_ARGS || log "WARNING: Verification failed — check 'make logs-device'"

log ""
log "=== Deployment complete ==="
log "  API: http://localhost:8088/v1"
log "  Container: ${DEVICE_PROJECT}-vllm-tt-1"
log ""
log "Commands (run from $DEPLOY_DIR/docker_tt):"
log "  make logs-device          # follow logs"
log "  make stop-device          # stop"
log "  make run-device           # restart (fast, C++ cached)"
log "  make verify-device        # check output"
log "  make diagnostics-device   # run diagnostics"
