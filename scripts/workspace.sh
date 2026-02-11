#!/bin/bash
# Workspace Manager for Coordinated Multi-Repo Development
# Manages bare repos and worktrees for docker_tt, tt-metal, and vllm
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_TT_DIR="$(dirname "$SCRIPT_DIR")"

# Load configuration
if [ -f "$SCRIPT_DIR/workspace.env" ]; then
    source "$SCRIPT_DIR/workspace.env"
else
    echo "Warning: workspace.env not found. Using defaults."
    echo "Run 'cp scripts/workspace.env.example scripts/workspace.env' and configure your forks."
fi

# Defaults
#
# WORKSPACE_BASE wants to be the directory that contains:
# - docker_tt.git / tt-metal.git / vllm.git (bare repos)
# - ws/ (worktrees)
#
# Historically docker_tt lived at "$WORKSPACE_BASE/docker_tt". With worktrees,
# docker_tt can also live at "$WORKSPACE_BASE/ws/<name>/docker_tt". Make the
# default robust so running this script from inside a worktree still "does the
# right thing" and targets the shared bare repos.
if [ -z "${WORKSPACE_BASE:-}" ]; then
    case "$DOCKER_TT_DIR" in
        */ws/*/docker_tt)
            WORKSPACE_BASE="${DOCKER_TT_DIR%%/ws/*}"
            ;;
        *)
            WORKSPACE_BASE="$(dirname "$DOCKER_TT_DIR")"
            ;;
    esac
fi
TT_METAL_DEFAULT_BRANCH="${TT_METAL_DEFAULT_BRANCH:-main}"
VLLM_DEFAULT_BRANCH="${VLLM_DEFAULT_BRANCH:-dev}"

# Bare repo paths
DOCKER_TT_BARE="$WORKSPACE_BASE/docker_tt.git"
TT_METAL_BARE="$WORKSPACE_BASE/tt-metal.git"
VLLM_BARE="$WORKSPACE_BASE/vllm.git"

# Workspace root
WS_ROOT="$WORKSPACE_BASE/ws"

#==============================================================================
# Helper Functions
#==============================================================================

log() {
    echo "[workspace] $*"
}

ensure_origin_fetch_refspec() {
    # Some bare repos can end up without a remote.origin.fetch refspec, which
    # prevents origin/* refs from being created/updated. Ensure a standard
    # "all branches" refspec exists.
    local repo_bare="$1"
    if ! git -C "$repo_bare" config --get-all remote.origin.fetch >/dev/null 2>&1; then
        git -C "$repo_bare" config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
    fi
}

error() {
    echo "[workspace] ERROR: $*" >&2
    exit 1
}

check_config() {
    local missing=()

    if [ -z "$DOCKER_TT_FORK" ] || [[ "$DOCKER_TT_FORK" == *"<"* ]]; then
        missing+=("DOCKER_TT_FORK")
    fi
    if [ -z "$TT_METAL_FORK" ] || [[ "$TT_METAL_FORK" == *"<"* ]]; then
        missing+=("TT_METAL_FORK")
    fi
    if [ -z "$VLLM_FORK" ] || [[ "$VLLM_FORK" == *"<"* ]]; then
        missing+=("VLLM_FORK")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        error "Missing or unconfigured variables in workspace.env: ${missing[*]}
Please edit scripts/workspace.env and set your fork URLs."
    fi
}

#==============================================================================
# init - Clone repos as bare and create main workspace
#==============================================================================

cmd_init() {
    check_config

    log "Initializing workspace infrastructure..."
    log "Base directory: $WORKSPACE_BASE"

    # Create bare repos
    mkdir -p "$WORKSPACE_BASE"

    # docker_tt bare repo
    if [ ! -d "$DOCKER_TT_BARE" ]; then
        log "Cloning docker_tt as bare repo..."
        git clone --bare "$DOCKER_TT_FORK" "$DOCKER_TT_BARE"
        ensure_origin_fetch_refspec "$DOCKER_TT_BARE"
        # No upstream for docker_tt (it IS the orchestration repo)
    else
        log "docker_tt.git already exists, skipping..."
        ensure_origin_fetch_refspec "$DOCKER_TT_BARE"
    fi

    # tt-metal bare repo
    if [ ! -d "$TT_METAL_BARE" ]; then
        log "Cloning tt-metal as bare repo (this may take a while)..."
        git clone --bare "$TT_METAL_FORK" "$TT_METAL_BARE"
        ensure_origin_fetch_refspec "$TT_METAL_BARE"

        # Add upstream remote
        if [ -n "$TT_METAL_UPSTREAM" ]; then
            log "Adding upstream remote for tt-metal..."
            git -C "$TT_METAL_BARE" remote add upstream "$TT_METAL_UPSTREAM" 2>/dev/null || true
        fi
    else
        log "tt-metal.git already exists, skipping..."
        ensure_origin_fetch_refspec "$TT_METAL_BARE"
    fi

    # vllm bare repo
    if [ ! -d "$VLLM_BARE" ]; then
        log "Cloning vllm as bare repo..."
        git clone --bare "$VLLM_FORK" "$VLLM_BARE"
        ensure_origin_fetch_refspec "$VLLM_BARE"

        # Add upstream remote
        if [ -n "$VLLM_UPSTREAM" ]; then
            log "Adding upstream remote for vllm..."
            git -C "$VLLM_BARE" remote add upstream "$VLLM_UPSTREAM" 2>/dev/null || true
        fi
    else
        log "vllm.git already exists, skipping..."
        ensure_origin_fetch_refspec "$VLLM_BARE"
    fi

    # Create main workspace
    log "Creating main workspace..."
    cmd_create "main" "$TT_METAL_DEFAULT_BRANCH" "$VLLM_DEFAULT_BRANCH"

    log ""
    log "Initialization complete!"
    log "Workspaces directory: $WS_ROOT"
    log ""
    log "Next steps:"
    log "  1. cd $WS_ROOT/main"
    log "  2. Create a feature workspace: make workspace-create NAME=my-feature"
}

#==============================================================================
# create - Create a new workspace with coordinated worktrees
#==============================================================================

cmd_create() {
    local name="$1"
    local tt_metal_branch="${2:-$TT_METAL_DEFAULT_BRANCH}"
    local vllm_branch="${3:-$VLLM_DEFAULT_BRANCH}"

    if [ -z "$name" ]; then
        error "Usage: workspace.sh create <name> [tt-metal-branch] [vllm-branch]"
    fi

    local ws_dir="$WS_ROOT/$name"

    if [ -d "$ws_dir" ]; then
        error "Workspace '$name' already exists at $ws_dir"
    fi

    # Verify bare repos exist
    if [ ! -d "$DOCKER_TT_BARE" ] || [ ! -d "$TT_METAL_BARE" ] || [ ! -d "$VLLM_BARE" ]; then
        error "Bare repos not found. Run 'workspace.sh init' first."
    fi

    log "Creating workspace '$name'..."
    mkdir -p "$ws_dir"

    # Create worktrees
    # For 'main' workspace, use existing branches
    # For feature workspaces, create new branches from current HEAD

    if [ "$name" = "main" ]; then
        # Main workspace: use default branches
        log "  Creating docker_tt worktree (branch: main)..."
        git -C "$DOCKER_TT_BARE" worktree add "$ws_dir/docker_tt" main 2>/dev/null || \
            git -C "$DOCKER_TT_BARE" worktree add "$ws_dir/docker_tt" -b main origin/main

        log "  Creating tt-metal worktree (branch: $tt_metal_branch)..."
        git -C "$TT_METAL_BARE" worktree add "$ws_dir/tt-metal" "$tt_metal_branch" 2>/dev/null || \
            git -C "$TT_METAL_BARE" worktree add "$ws_dir/tt-metal" -b "$tt_metal_branch" "origin/$tt_metal_branch"

        log "  Creating vllm worktree (branch: $vllm_branch)..."
        git -C "$VLLM_BARE" worktree add "$ws_dir/vllm" "$vllm_branch" 2>/dev/null || \
            git -C "$VLLM_BARE" worktree add "$ws_dir/vllm" -b "$vllm_branch" "origin/$vllm_branch"
    else
        # Feature workspace: check if remote branches exist, otherwise create new ones.
        # Fetch first so we see any recently-pushed branches.
        log "  Fetching latest refs..."
        git -C "$DOCKER_TT_BARE" fetch origin --prune 2>/dev/null || true
        git -C "$TT_METAL_BARE" fetch origin --prune 2>/dev/null || true
        git -C "$VLLM_BARE" fetch origin --prune 2>/dev/null || true

        # docker_tt: check for origin/<name>, fall back to new branch from main
        if git -C "$DOCKER_TT_BARE" rev-parse --verify "origin/$name" &>/dev/null; then
            log "  Creating docker_tt worktree (existing remote branch: $name)..."
            git -C "$DOCKER_TT_BARE" worktree add "$ws_dir/docker_tt" "$name" 2>/dev/null || \
                git -C "$DOCKER_TT_BARE" worktree add "$ws_dir/docker_tt" -b "$name" "origin/$name"
            git -C "$ws_dir/docker_tt" branch --set-upstream-to="origin/$name" "$name" 2>/dev/null || true
        else
            log "  Creating docker_tt worktree (new branch: $name from main)..."
            git -C "$DOCKER_TT_BARE" worktree add -b "$name" "$ws_dir/docker_tt" main
        fi

        # tt-metal: check for origin/<name>, fall back to new branch from default
        if git -C "$TT_METAL_BARE" rev-parse --verify "origin/$name" &>/dev/null; then
            log "  Creating tt-metal worktree (existing remote branch: $name)..."
            git -C "$TT_METAL_BARE" worktree add "$ws_dir/tt-metal" "$name" 2>/dev/null || \
                git -C "$TT_METAL_BARE" worktree add "$ws_dir/tt-metal" -b "$name" "origin/$name"
            git -C "$ws_dir/tt-metal" branch --set-upstream-to="origin/$name" "$name" 2>/dev/null || true
        else
            log "  Creating tt-metal worktree (new branch: $name from origin/$tt_metal_branch)..."
            git -C "$TT_METAL_BARE" worktree add -b "$name" "$ws_dir/tt-metal" "origin/$tt_metal_branch"
        fi

        # vllm: check for origin/<name>, fall back to new branch from default
        if git -C "$VLLM_BARE" rev-parse --verify "origin/$name" &>/dev/null; then
            log "  Creating vllm worktree (existing remote branch: $name)..."
            git -C "$VLLM_BARE" worktree add "$ws_dir/vllm" "$name" 2>/dev/null || \
                git -C "$VLLM_BARE" worktree add "$ws_dir/vllm" -b "$name" "origin/$name"
            git -C "$ws_dir/vllm" branch --set-upstream-to="origin/$name" "$name" 2>/dev/null || true
        else
            log "  Creating vllm worktree (new branch: $name from origin/$vllm_branch)..."
            git -C "$VLLM_BARE" worktree add -b "$name" "$ws_dir/vllm" "origin/$vllm_branch"
        fi
    fi

    log ""
    log "Workspace '$name' created at: $ws_dir"
    log "  docker_tt: $ws_dir/docker_tt"
    log "  tt-metal:  $ws_dir/tt-metal"
    log "  vllm:      $ws_dir/vllm"
}

#==============================================================================
# list - List all workspaces
#==============================================================================

cmd_list() {
    if [ ! -d "$WS_ROOT" ]; then
        log "No workspaces found. Run 'workspace.sh init' first."
        return 0
    fi

    log "Workspaces in $WS_ROOT:"
    echo ""

    for ws in "$WS_ROOT"/*/; do
        if [ -d "$ws" ]; then
            local name=$(basename "$ws")
            echo "  $name/"

            # Show branch info for each repo
            for repo in docker_tt tt-metal vllm; do
                if [ -d "$ws/$repo" ]; then
                    local branch=$(git -C "$ws/$repo" branch --show-current 2>/dev/null || echo "detached")
                    echo "    $repo: $branch"
                fi
            done
            echo ""
        fi
    done
}

#==============================================================================
# delete - Remove a workspace and its worktrees
#==============================================================================

cmd_delete() {
    local name="$1"

    if [ -z "$name" ]; then
        error "Usage: workspace.sh delete <name>"
    fi

    if [ "$name" = "main" ]; then
        error "Cannot delete the 'main' workspace"
    fi

    local ws_dir="$WS_ROOT/$name"

    if [ ! -d "$ws_dir" ]; then
        error "Workspace '$name' not found at $ws_dir"
    fi

    log "Deleting workspace '$name'..."

    # Remove worktrees properly
    for repo_bare in "$DOCKER_TT_BARE" "$TT_METAL_BARE" "$VLLM_BARE"; do
        local repo_name=$(basename "$repo_bare" .git)
        local worktree_path="$ws_dir/$repo_name"

        if [ -d "$worktree_path" ]; then
            log "  Removing $repo_name worktree..."
            git -C "$repo_bare" worktree remove "$worktree_path" --force 2>/dev/null || true
        fi
    done

    # Remove workspace directory if still exists
    if [ -d "$ws_dir" ]; then
        rm -rf "$ws_dir"
    fi

    # Optionally delete the branches (ask user)
    log ""
    log "Workspace '$name' deleted."
    log "Note: Feature branches named '$name' still exist in bare repos."
    log "To delete them: git -C <repo>.git branch -D $name"
}

#==============================================================================
# status - Show git status across all repos in a workspace
#==============================================================================

cmd_status() {
    local name="${1:-main}"
    local ws_dir="$WS_ROOT/$name"

    if [ ! -d "$ws_dir" ]; then
        error "Workspace '$name' not found"
    fi

    log "Status for workspace '$name':"
    echo ""

    for repo in docker_tt tt-metal vllm; do
        local repo_path="$ws_dir/$repo"
        if [ -d "$repo_path" ]; then
            echo "=== $repo ==="
            local branch=$(git -C "$repo_path" branch --show-current 2>/dev/null || echo "detached")
            echo "Branch: $branch"
            git -C "$repo_path" status -s
            echo ""
        fi
    done
}

#==============================================================================
# sync - Fetch from origin and upstream for all repos
#==============================================================================

cmd_sync() {
    log "Syncing all bare repos..."

    for repo_bare in "$DOCKER_TT_BARE" "$TT_METAL_BARE" "$VLLM_BARE"; do
        if [ -d "$repo_bare" ]; then
            local repo_name=$(basename "$repo_bare" .git)
            log "Fetching $repo_name..."

            ensure_origin_fetch_refspec "$repo_bare"
            git -C "$repo_bare" fetch origin --prune 2>/dev/null || \
                log "  Warning: Failed to fetch from origin"

            git -C "$repo_bare" fetch upstream --prune 2>/dev/null || \
                true  # upstream may not exist for docker_tt
        fi
    done

    log "Sync complete."
}

#==============================================================================
# path - Output the path to a workspace (for use in scripts)
#==============================================================================

cmd_path() {
    local name="${1:-main}"
    local ws_dir="$WS_ROOT/$name"

    if [ ! -d "$ws_dir" ]; then
        error "Workspace '$name' not found"
    fi

    echo "$ws_dir"
}

#==============================================================================
# Main
#==============================================================================

usage() {
    cat <<EOF
Workspace Manager for Coordinated Multi-Repo Development

Usage: workspace.sh <command> [args]

Commands:
  init                    Clone all repos as bare, add remotes, create main workspace
  create <name>           Create a new workspace with coordinated worktrees
  list                    List all workspaces
  delete <name>           Remove a workspace and its worktrees
  status [name]           Show git status across all repos (default: main)
  sync                    Fetch from origin and upstream for all repos
  path [name]             Output workspace path (for scripts)

Configuration:
  Edit scripts/workspace.env to set your fork URLs.

Examples:
  workspace.sh init                    # First-time setup
  workspace.sh create fix-memory-leak  # Create feature workspace
  workspace.sh status fix-memory-leak  # Check status
  workspace.sh delete fix-memory-leak  # Clean up
EOF
}

case "${1:-}" in
    init)    cmd_init ;;
    create)  shift; cmd_create "$@" ;;
    list)    cmd_list ;;
    delete)  shift; cmd_delete "$@" ;;
    status)  shift; cmd_status "$@" ;;
    sync)    cmd_sync ;;
    path)    shift; cmd_path "$@" ;;
    -h|--help|help|"")
        usage
        ;;
    *)
        error "Unknown command: $1. Run 'workspace.sh --help' for usage."
        ;;
esac
