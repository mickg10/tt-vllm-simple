#!/usr/bin/env bash
# deploy-galaxy.sh — Deploy GLM-4.7-Flash on a Galaxy Wormhole machine
#
# Usage:
#   ./deploy-galaxy.sh [--deploy-dir /path/to/deploy]
#
# This script:
#   1. Clones tt-metal, vllm, docker_tt from GitHub fork (galaxy_wormhole branch)
#   2. Builds the Docker dev image
#   3. Starts the container (builds C++, installs Python, loads model)
#   4. Waits for health check
#   5. Verifies coherent output
#
# Prerequisites on the Galaxy machine:
#   - Docker + docker compose v2
#   - Access to /dev/tenstorrent (TT devices)
#   - /dev/hugepages-1G mounted
#   - ~/.cache/huggingface populated with zai-org/GLM-4.7-Flash (or HF_TOKEN set)
#   - `sg docker` access (user in docker group)
#
# The script is idempotent: re-running it will skip already-completed steps.
set -euo pipefail

#=============================================================================
# Configuration (override with environment variables or CLI flags)
#=============================================================================
DEPLOY_DIR="${DEPLOY_DIR:-/home/user/src_docker/ws/glm47_flash_deploy}"
BRANCH="${BRANCH:-galaxy_wormhole}"
TT_METAL_REPO="${TT_METAL_REPO:-https://github.com/mickg10/tt-metal.git}"
VLLM_REPO="${VLLM_REPO:-https://github.com/mickg10/vllm.git}"
DOCKER_TT_REPO="${DOCKER_TT_REPO:-https://github.com/mickg10/tt-vllm-simple.git}"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-glm-flash-deploy}"
BUILD_JOBS="${BUILD_JOBS:-32}"

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --deploy-dir) DEPLOY_DIR="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --project-name) COMPOSE_PROJECT_NAME="$2"; shift 2 ;;
        --build-jobs) BUILD_JOBS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--deploy-dir DIR] [--branch BRANCH] [--project-name NAME] [--build-jobs N]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

ENV_FILE="dev/.env.glm47.galaxy"
COMPOSE_FILES="-f dev/docker-compose.yml -f dev/docker-compose.galaxy.yml"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "FATAL: $*" >&2; exit 1; }

#=============================================================================
# Step 0: Validate environment
#=============================================================================
log "=== GLM-4.7-Flash Galaxy Deploy ==="
log "Deploy dir: $DEPLOY_DIR"
log "Branch: $BRANCH"
log "Project: $COMPOSE_PROJECT_NAME"

command -v docker >/dev/null 2>&1 || die "docker not found"
command -v git >/dev/null 2>&1 || die "git not found"
docker compose version >/dev/null 2>&1 || die "docker compose v2 not found"

if [ ! -d /dev/hugepages-1G ]; then
    log "WARNING: /dev/hugepages-1G not found — container may fail to start"
fi

#=============================================================================
# Step 1: Clone repositories
#=============================================================================
mkdir -p "$DEPLOY_DIR"

clone_repo() {
    local repo_url="$1"
    local target_dir="$2"
    local branch="$3"

    if [ -d "$target_dir/.git" ]; then
        log "  $target_dir already exists, fetching latest..."
        cd "$target_dir"
        git fetch origin "$branch" --depth=1 2>/dev/null || git fetch origin "$branch"
        git checkout "$branch" 2>/dev/null || git checkout -b "$branch" "origin/$branch"
        git reset --hard "origin/$branch"
        cd - >/dev/null
    else
        log "  Cloning $repo_url → $target_dir (branch: $branch)..."
        git clone --depth=1 --branch "$branch" "$repo_url" "$target_dir"
    fi
}

log "Step 1: Cloning repositories..."
clone_repo "$TT_METAL_REPO" "$DEPLOY_DIR/tt-metal" "$BRANCH"
clone_repo "$VLLM_REPO" "$DEPLOY_DIR/vllm" "$BRANCH"
clone_repo "$DOCKER_TT_REPO" "$DEPLOY_DIR/docker_tt" "$BRANCH"

# Initialize required submodules (tt-metal needs these for the build)
# Clean stale root-owned tracy dir that Docker may have created via bind mount
if [ -d "$DEPLOY_DIR/tt-metal/tt_metal/third_party/tracy" ] && \
   [ ! -f "$DEPLOY_DIR/tt-metal/tt_metal/third_party/tracy/CMakeLists.txt" ]; then
    log "  Cleaning stale tracy submodule directory..."
    sudo rm -rf "$DEPLOY_DIR/tt-metal/tt_metal/third_party/tracy" 2>/dev/null || \
        rm -rf "$DEPLOY_DIR/tt-metal/tt_metal/third_party/tracy" 2>/dev/null || true
fi
log "  Initializing tt-metal submodules..."
cd "$DEPLOY_DIR/tt-metal"
git submodule update --init --depth=1 \
    tt_metal/third_party/tt_llk \
    tt_metal/third_party/umd \
    tt_metal/third_party/tracy 2>/dev/null || \
    git submodule update --init \
        tt_metal/third_party/tt_llk \
        tt_metal/third_party/umd \
        tt_metal/third_party/tracy
cd - >/dev/null

#=============================================================================
# Step 2: Patch env file with correct WORKSPACE_PATH
#=============================================================================
log "Step 2: Configuring environment..."

cd "$DEPLOY_DIR/docker_tt"

# Update WORKSPACE_PATH in env file to point to this deploy directory
if grep -q "^WORKSPACE_PATH=" "$ENV_FILE"; then
    sed -i "s|^WORKSPACE_PATH=.*|WORKSPACE_PATH=$DEPLOY_DIR|" "$ENV_FILE"
else
    echo "WORKSPACE_PATH=$DEPLOY_DIR" >> "$ENV_FILE"
fi

# First deploy: need to build C++ (no cached .so yet)
# Set SKIP_TT_METAL_BUILD=0 for initial build
sed -i 's/^SKIP_TT_METAL_BUILD=1/SKIP_TT_METAL_BUILD=0/' "$ENV_FILE"

# Set BUILD_JOBS
sed -i "s/^BUILD_JOBS=.*/BUILD_JOBS=$BUILD_JOBS/" "$ENV_FILE"

log "  Env file: $ENV_FILE"
log "  WORKSPACE_PATH=$DEPLOY_DIR"
log "  SKIP_TT_METAL_BUILD=0 (will build C++ on first run)"

#=============================================================================
# Step 3: Build Docker image
#=============================================================================
log "Step 3: Building Docker dev image..."

sg docker -c "docker compose --env-file $ENV_FILE $COMPOSE_FILES \
    -p $COMPOSE_PROJECT_NAME build vllm-tt" 2>&1 | tail -5

log "  Docker image built."

#=============================================================================
# Step 4: Start container
#=============================================================================
log "Step 4: Starting container..."
log "  This will: build tt-metal C++ (~15-30 min), install Python deps, load model weights, warm up."
log "  Total first-run time: ~30-60 minutes."

sg docker -c "docker compose --env-file $ENV_FILE $COMPOSE_FILES \
    -p $COMPOSE_PROJECT_NAME up -d vllm-tt"

CONTAINER_NAME="${COMPOSE_PROJECT_NAME}-vllm-tt-1"
log "  Container: $CONTAINER_NAME"

#=============================================================================
# Step 5: Wait for health check
#=============================================================================
log "Step 5: Waiting for container to become healthy..."
log "  (Timeout: 90 minutes for first run with C++ build + weight loading)"

MAX_WAIT=5400  # 90 minutes
INTERVAL=30
elapsed=0

while [ $elapsed -lt $MAX_WAIT ]; do
    status=$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "not_found")

    case "$status" in
        healthy)
            log "  Container healthy after ${elapsed}s!"
            break
            ;;
        unhealthy)
            log "  Container unhealthy — checking logs..."
            docker logs --tail 20 "$CONTAINER_NAME" 2>&1
            die "Container became unhealthy"
            ;;
        not_found)
            die "Container $CONTAINER_NAME not found"
            ;;
        *)
            # Show progress every 5 minutes
            if (( elapsed % 300 == 0 )) && (( elapsed > 0 )); then
                log "  Still starting... (${elapsed}s elapsed, status: $status)"
                docker logs --tail 3 "$CONTAINER_NAME" 2>&1 | head -3
            fi
            ;;
    esac

    sleep $INTERVAL
    elapsed=$((elapsed + INTERVAL))
done

if [ $elapsed -ge $MAX_WAIT ]; then
    log "  Timed out after ${MAX_WAIT}s. Last logs:"
    docker logs --tail 30 "$CONTAINER_NAME" 2>&1
    die "Container did not become healthy within ${MAX_WAIT}s"
fi

#=============================================================================
# Step 6: Set SKIP_TT_METAL_BUILD=1 for future restarts
#=============================================================================
sed -i 's/^SKIP_TT_METAL_BUILD=0/SKIP_TT_METAL_BUILD=1/' "$ENV_FILE"
log "  Set SKIP_TT_METAL_BUILD=1 for future restarts"

#=============================================================================
# Step 7: Verify coherent output
#=============================================================================
log "Step 7: Verifying coherent output..."

RESPONSE=$(curl -s http://localhost:8088/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"zai-org/GLM-4.7-Flash","prompt":"Hello, how are you today?","max_tokens":30,"temperature":0}')

if echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); t=d['choices'][0]['text']; print(t); assert len(t.split())>3" 2>/dev/null; then
    log "  Output looks coherent!"
else
    log "  WARNING: Output may be garbled:"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
fi

#=============================================================================
# Done
#=============================================================================
log ""
log "=== Deployment complete ==="
log "  API endpoint: http://localhost:8088/v1"
log "  Container:    $CONTAINER_NAME"
log "  Deploy dir:   $DEPLOY_DIR"
log ""
log "Useful commands:"
log "  # View logs:"
log "  docker logs -f $CONTAINER_NAME"
log ""
log "  # Run benchmark:"
log "  docker exec $CONTAINER_NAME bash -c 'cd /vllm && python tests/bench_decode.py --gen-tokens 50 --only-batch 1 --skip-combined --prefill-contexts 0'"
log ""
log "  # Stop:"
log "  cd $DEPLOY_DIR/docker_tt && sg docker -c 'docker compose --env-file $ENV_FILE $COMPOSE_FILES -p $COMPOSE_PROJECT_NAME down'"
log ""
log "  # Restart (fast — C++ build cached):"
log "  cd $DEPLOY_DIR/docker_tt && sg docker -c 'docker compose --env-file $ENV_FILE $COMPOSE_FILES -p $COMPOSE_PROJECT_NAME up -d vllm-tt'"
log ""
log "  # Remove deployment (Docker creates root-owned files, need sudo):"
log "  cd $DEPLOY_DIR/docker_tt && sg docker -c 'docker compose --env-file $ENV_FILE $COMPOSE_FILES -p $COMPOSE_PROJECT_NAME down -v'"
log "  sudo rm -rf $DEPLOY_DIR"
