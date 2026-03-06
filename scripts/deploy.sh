#!/usr/bin/env bash
# deploy.sh — Deploy or diagnose GLM-4.7-Flash on any TT device
#
# Device-agnostic replacement for deploy-galaxy.sh. Derives all device-specific
# behavior (reset command, compose overrides, cluster type) from MESH_DEVICE
# in the env file.
#
# Modes:
#   run      (default) Clone repos, build, start vLLM, wait for healthy, verify
#   diagnostics Clone repos, reset devices, run tt-triage (no vLLM)
#   shell       Clone repos, drop into interactive shell with tt-tools (no vLLM)
#
# Usage:
#   ./deploy.sh --device galaxy                    # Galaxy, default mode (run)
#   ./deploy.sh --device t3k --mode diagnostics    # T3K diagnostics
#   ./deploy.sh --device galaxy --mode shell       # Galaxy interactive shell
#   ./deploy.sh --env-file dev/.env.glm47.galaxy   # explicit env file
#
# Prerequisites:
#   - Docker access (docker compose must work)
#   - /dev/tenstorrent (TT devices)
#   - /dev/hugepages-1G mounted (run mode)
#   - ~/.cache/huggingface with model weights or HF_TOKEN set (run mode)
#
# Idempotent: re-running skips already-completed steps.
set -euo pipefail

#=============================================================================
# Device helpers — derive everything from MESH_DEVICE
#=============================================================================

# Map MESH_DEVICE value to human-friendly slug
mesh_to_slug() {
    case "$1" in
        TG)   echo "galaxy" ;;
        T3K)  echo "t3k" ;;
        N300) echo "n300" ;;
        N150) echo "n150" ;;
        BH)   echo "blackhole" ;;
        *)    echo "$1" ;;
    esac
}

# Map slug to MESH_DEVICE value (reverse of mesh_to_slug)
slug_to_mesh() {
    case "$1" in
        galaxy)    echo "TG" ;;
        t3k)       echo "T3K" ;;
        n300)      echo "N300" ;;
        n150)      echo "N150" ;;
        blackhole) echo "BH" ;;
        *)         echo "$1" ;;
    esac
}

# Determine compose override file (if it exists)
resolve_compose_extra() {
    local slug="$1"
    local compose_file="dev/docker-compose.${slug}.yml"
    if [ -f "$compose_file" ]; then
        echo "-f $compose_file"
    fi
}

# Device reset command (Galaxy needs -glx_reset, everything else uses -r)
reset_devices() {
    local mesh_device="$1"
    case "$mesh_device" in
        TG)
            tt-smi -glx_reset 2>/dev/null || tt-smi -r 2>/dev/null || \
                log "WARNING: Device reset not available"
            ;;
        *)
            tt-smi -r 2>/dev/null || \
                log "WARNING: Device reset not available"
            ;;
    esac
    rm -f /dev/shm/TT_UMD_LOCK.* 2>/dev/null || true
}

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
DEVICE=""
ENV_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --env-file) ENV_FILE="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --deploy-dir) DEPLOY_DIR="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --project-name) DEVICE_PROJECT="$2"; shift 2 ;;
        --build-jobs) BUILD_JOBS="$2"; shift 2 ;;
        -h|--help)
            cat <<'USAGE'
Usage: deploy.sh [OPTIONS]

Device selection (one required):
  --device SLUG          Device slug: galaxy, t3k, n300, n150, blackhole
                         Finds dev/.env.glm47.<slug> by convention
  --env-file PATH        Explicit env file path (reads MESH_DEVICE from file)

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
  # Galaxy deploy (default):
  ./deploy.sh --device galaxy

  # T3K diagnostics:
  ./deploy.sh --device t3k --mode diagnostics

  # Galaxy interactive shell:
  ./deploy.sh --device galaxy --mode shell

  # Explicit env file:
  ./deploy.sh --env-file dev/.env.glm47.galaxy

  # Deploy to custom directory:
  ./deploy.sh --device galaxy --deploy-dir /home/user/my_workspace

  # Backward compatible (via deploy-galaxy.sh wrapper):
  ./deploy-galaxy.sh --mode diagnostics
USAGE
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

#=============================================================================
# Resolve device → env file → MESH_DEVICE → slug → compose extras
#=============================================================================

if [ -n "$DEVICE" ] && [ -n "$ENV_FILE" ]; then
    die "Specify --device OR --env-file, not both"
fi

if [ -z "$DEVICE" ] && [ -z "$ENV_FILE" ]; then
    echo "FATAL: Must specify --device <slug> or --env-file <path>" >&2
    echo "Run with --help for usage." >&2
    exit 1
fi

# Resolve env file from device slug
if [ -n "$DEVICE" ]; then
    ENV_FILE="dev/.env.glm47.${DEVICE}"
fi

# Read MESH_DEVICE from env file (need it for compose extras and reset commands)
# We do this early so the clone step can proceed, but the file may not exist yet
# (it's inside the docker_tt repo we're about to clone). We'll re-read after clone.
MESH_DEVICE=""
DEVICE_SLUG=""

resolve_device_from_env() {
    if [ -f "$DEPLOY_DIR/docker_tt/$ENV_FILE" ]; then
        MESH_DEVICE=$(grep -m1 '^MESH_DEVICE=' "$DEPLOY_DIR/docker_tt/$ENV_FILE" | cut -d= -f2)
    fi
    if [ -n "$MESH_DEVICE" ]; then
        DEVICE_SLUG=$(mesh_to_slug "$MESH_DEVICE")
    elif [ -n "$DEVICE" ]; then
        DEVICE_SLUG="$DEVICE"
        MESH_DEVICE=$(slug_to_mesh "$DEVICE")
    fi
}

case "$MODE" in
    run|diagnostics|shell) ;;
    *) echo "FATAL: Unknown mode '$MODE'. Use 'run', 'diagnostics', or 'shell'."; exit 1 ;;
esac

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "FATAL: $*" >&2; exit 1; }

#=============================================================================
# Step 0: Validate environment
#=============================================================================
log "=== GLM-4.7-Flash Deploy — device: ${DEVICE:-custom}, mode: $MODE ==="
log "Workspace: $DEPLOY_DIR"
log "Branch: $BRANCH"
log "Env file: $ENV_FILE"

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

# Delete non-semver tags that break setuptools_scm (vllm pip install)
# and ensure at least one semver tag exists for CMake version detection
log "  Cleaning non-semver tags..."
for repo_dir in "$DEPLOY_DIR/vllm" "$DEPLOY_DIR/tt-metal"; do
    (
        cd "$repo_dir"
        for tag in $(git tag -l); do
            # Keep tags matching vN.N.N (semver) — delete everything else
            if ! echo "$tag" | grep -qE '^v[0-9]+\.[0-9]+\.[0-9]+'; then
                git tag -d "$tag" >/dev/null 2>&1 && log "    Deleted tag: $tag ($(basename "$repo_dir"))"
            fi
        done
        # Shallow clones may have no tags at all — CMake needs git describe to work
        if ! git describe --tags --first-parent >/dev/null 2>&1; then
            git tag -a v0.0.0 -m "placeholder for shallow clone build" 2>/dev/null && \
                log "    Created v0.0.0 tag ($(basename "$repo_dir"), shallow clone)"
        fi
    )
done

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

if [ ! -f "$ENV_FILE" ] && [ -n "$DEVICE" ]; then
    # Convention-named file doesn't exist; scan dev/.env.glm47* for matching MESH_DEVICE
    EXPECTED_MESH=$(slug_to_mesh "$DEVICE")
    FOUND_ENV=""
    for f in dev/.env.glm47*; do
        [ -f "$f" ] || continue
        if grep -q "^MESH_DEVICE=${EXPECTED_MESH}\$" "$f" 2>/dev/null; then
            FOUND_ENV="$f"
            break
        fi
    done
    if [ -n "$FOUND_ENV" ]; then
        log "  Convention file $ENV_FILE not found, using $FOUND_ENV (has MESH_DEVICE=$EXPECTED_MESH)"
        ENV_FILE="$FOUND_ENV"
    else
        die "Env file not found: $ENV_FILE (and no dev/.env.glm47* has MESH_DEVICE=$EXPECTED_MESH)"
    fi
elif [ ! -f "$ENV_FILE" ]; then
    die "Env file not found: $ENV_FILE — check --device or --env-file argument"
fi

# Now that we have the env file, resolve device info
resolve_device_from_env

sed -i "s|^WORKSPACE_PATH=.*|WORKSPACE_PATH=$DEPLOY_DIR|" "$ENV_FILE"

# Build compose command with optional device-specific override
COMPOSE_EXTRA=$(resolve_compose_extra "$DEVICE_SLUG")
COMPOSE_BASE="docker compose --env-file $ENV_FILE -f dev/docker-compose.yml${COMPOSE_EXTRA:+ $COMPOSE_EXTRA}"

log "  MESH_DEVICE=$MESH_DEVICE (slug: $DEVICE_SLUG)"
log "  Compose: $COMPOSE_BASE"

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
    reset_devices "$MESH_DEVICE"

    log ""
    log "Step 5: Running diagnostics (foreground)..."
    $COMPOSE_BASE -f dev/docker-compose.diagnostics.yml run --rm vllm-tt

    log ""
    log "Step 6: Resetting devices (clean state)..."
    reset_devices "$MESH_DEVICE"

    log ""
    log "=== Diagnostics complete. Devices reset. ==="
    log "To start vLLM: cd $DEPLOY_DIR/docker_tt && make run-device DEVICE=$DEVICE_SLUG"
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
    log "To start vLLM: cd $DEPLOY_DIR/docker_tt && make run-device DEVICE=$DEVICE_SLUG"
    exit 0
fi

#=============================================================================
# Mode: run
#=============================================================================
sed -i 's/^SKIP_TT_METAL_BUILD=1/SKIP_TT_METAL_BUILD=0/' "$ENV_FILE"
sed -i "s/^BUILD_JOBS=.*/BUILD_JOBS=$BUILD_JOBS/" "$ENV_FILE"

log "  WORKSPACE_PATH=$DEPLOY_DIR"
log "  SKIP_TT_METAL_BUILD=0 (first run builds C++)"

MAKE_ARGS="DEVICE_PROJECT=$DEVICE_PROJECT DEVICE=$DEVICE_SLUG"

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
log "  make logs-device DEVICE=$DEVICE_SLUG          # follow logs"
log "  make stop-device DEVICE=$DEVICE_SLUG          # stop"
log "  make run-device DEVICE=$DEVICE_SLUG           # restart (fast, C++ cached)"
log "  make verify-device DEVICE=$DEVICE_SLUG        # check output"
log "  make diagnostics-device DEVICE=$DEVICE_SLUG   # run diagnostics"
