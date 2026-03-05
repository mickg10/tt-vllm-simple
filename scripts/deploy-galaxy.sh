#!/usr/bin/env bash
# deploy-galaxy.sh — Deploy GLM-4.7-Flash on a Galaxy Wormhole machine
#
# Usage:
#   ./deploy-galaxy.sh [--deploy-dir /path/to/deploy]
#
# This script:
#   1. Clones tt-metal, vllm, docker_tt from GitHub fork (galaxy_wormhole branch)
#   2. Runs `make run-galaxy` (builds image, starts container)
#   3. Runs `make wait-galaxy` (blocks until healthy)
#   4. Runs `make verify-galaxy` (checks coherent output)
#
# Prerequisites:
#   - Docker access (docker ps must work — any group: docker, vidocker, etc.)
#   - /dev/tenstorrent (TT devices)
#   - /dev/hugepages-1G mounted
#   - ~/.cache/huggingface with zai-org/GLM-4.7-Flash (or HF_TOKEN set)
#
# The script is idempotent: re-running it will skip already-completed steps.
set -euo pipefail

#=============================================================================
# Configuration (override with environment variables or CLI flags)
#=============================================================================
DEPLOY_DIR="${DEPLOY_DIR:-${HOME}/src_docker/ws/glm47_flash_deploy}"
BRANCH="${BRANCH:-galaxy_wormhole}"
TT_METAL_REPO="${TT_METAL_REPO:-https://github.com/mickg10/tt-metal.git}"
VLLM_REPO="${VLLM_REPO:-https://github.com/mickg10/vllm.git}"
DOCKER_TT_REPO="${DOCKER_TT_REPO:-https://github.com/mickg10/tt-vllm-simple.git}"
DEVICE_PROJECT="${DEVICE_PROJECT:-glm-flash}"
BUILD_JOBS="${BUILD_JOBS:-32}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --deploy-dir) DEPLOY_DIR="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --project-name) DEVICE_PROJECT="$2"; shift 2 ;;
        --build-jobs) BUILD_JOBS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--deploy-dir DIR] [--branch BRANCH] [--project-name NAME] [--build-jobs N]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "FATAL: $*" >&2; exit 1; }

#=============================================================================
# Step 0: Validate environment
#=============================================================================
log "=== GLM-4.7-Flash Galaxy Deploy ==="
log "Deploy dir: $DEPLOY_DIR"
log "Branch: $BRANCH"

command -v git >/dev/null 2>&1 || die "git not found"
docker ps >/dev/null 2>&1 || die "Cannot run 'docker ps'. Check Docker group membership."

if [ ! -d /dev/hugepages-1G ]; then
    log "WARNING: /dev/hugepages-1G not found — container may fail to start"
fi

#=============================================================================
# Step 1: Clone repositories
#=============================================================================
mkdir -p "$DEPLOY_DIR"

clone_repo() {
    local repo_url="$1" target_dir="$2" branch="$3"
    if [ -d "$target_dir/.git" ]; then
        log "  $target_dir exists, fetching..."
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
sed -i 's/^SKIP_TT_METAL_BUILD=1/SKIP_TT_METAL_BUILD=0/' "$ENV_FILE"
sed -i "s/^BUILD_JOBS=.*/BUILD_JOBS=$BUILD_JOBS/" "$ENV_FILE"

log "  WORKSPACE_PATH=$DEPLOY_DIR"
log "  SKIP_TT_METAL_BUILD=0 (first run builds C++)"

#=============================================================================
# Step 3-5: Build, start, wait, verify — all via Makefile
#=============================================================================
MAKE_ARGS="DEVICE_PROJECT=$DEVICE_PROJECT"

log "Step 3: Building + starting (make run-device)..."
make run-device $MAKE_ARGS

log "Step 4: Waiting for healthy (make wait-device)..."
make wait-device $MAKE_ARGS

# Future restarts skip C++ build
sed -i 's/^SKIP_TT_METAL_BUILD=0/SKIP_TT_METAL_BUILD=1/' "$ENV_FILE"

log "Step 5: Verifying output (make verify-device)..."
make verify-device $MAKE_ARGS || log "WARNING: Verification failed — check 'make logs-device'"

#=============================================================================
# Done
#=============================================================================
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
