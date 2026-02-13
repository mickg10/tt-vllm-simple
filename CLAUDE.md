# TT vLLM Docker Setup

Docker setup for running vLLM with Tenstorrent hardware acceleration.

## Project Structure

```
docker_tt/
├── Makefile                 # Build, test, run commands
├── prebuilt/                # Fast startup using pre-built image
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── .env.example
├── from_source/             # Full source build (for development)
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── .env.example
├── dev/                     # Local source development (bind mounts)
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── dev-entrypoint.sh
│   └── .env.example
├── base/                    # Shared services (WebUI, Monitor)
│   └── docker-compose.yml
├── tt-monitor/              # Device monitoring dashboard
├── tests/                   # Test scripts
└── scripts/                 # Utility scripts
    ├── workspace.sh         # Workspace manager
    └── workspace.env.example
```

## Quick Start

```bash
# Using prebuilt image (recommended for most users)
make build-prebuilt
make run-prebuilt

# Or using source build (for development/customization)
make build-source    # Takes ~45 minutes first time
make run-source
```

## Services & Ports

| Service | Port | URL |
|---------|------|-----|
| vLLM API | 8088 | http://localhost:8088/v1 |
| Open WebUI | 3000 | http://localhost:3000 |
| TT Monitor | 9090 | http://localhost:9090 |

## Makefile Targets

```bash
make help            # Show all targets

# Build
make build-prebuilt  # Build prebuilt variant
make build-source    # Build from source (parallel, ~45min)

# Run
make run-prebuilt    # Start prebuilt stack
make run-source      # Start source stack

# Test (builds, starts, tests, stops)
make test-prebuilt   # Full test cycle for prebuilt
make test-source     # Full test cycle for source

# Stop
make stop-prebuilt   # Stop prebuilt stack
make stop-source     # Stop source stack

# Utilities
make logs-prebuilt   # Follow prebuilt logs
make logs-source     # Follow source logs
make reset-devices   # Reset TT devices
make clean           # Remove containers and images
```

## Configuration

Each variant has its own `.env.example` file. Copy to `.env` and customize:

```bash
cd prebuilt        # or from_source
cp .env.example .env
# Edit .env as needed
```

### Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| HF_MODEL | Qwen/Qwen3-0.6B | Model to serve |
| HF_TOKEN | (empty) | HuggingFace token for gated models |
| MESH_DEVICE | N300 | Device config: N150, N300, T3K, TG |
| BUILD_JOBS | 16 | Parallel jobs for source build |

## Build Variants

| Aspect | Prebuilt | Source | Dev |
|--------|----------|--------|-----|
| Build time | ~2 min | ~45 min | ~5 min |
| Image size | ~15 GB | ~25 GB | ~10 GB |
| Customization | Limited | Full | Full |
| Distributed | No | Yes | Yes |
| Code changes | Rebuild | Rebuild | Instant (Python) |
| Use case | Production | CI/Testing | Active development |

## Troubleshooting

### Device initialization failed
```bash
make reset-devices
make run-prebuilt  # or run-source
```

### Model download issues
Ensure HF_TOKEN is set for gated models:
```bash
echo "HF_TOKEN=hf_xxxx" >> prebuilt/.env
```

### Port already in use
Stop any existing stacks:
```bash
make stop-prebuilt
make stop-source
```

## Agentic Development Workflow

For coordinated development across docker_tt, tt-metal, and vllm repositories by AI agents (Claude, Codex, etc.) or human developers.

### Why Worktrees?

This setup uses git worktrees instead of submodules because:
- **AI agents can work independently** - Each repo has its own git history; no submodule SHA coordination
- **Parallel workspaces** - Multiple feature branches can exist simultaneously without conflicts
- **Clean commits** - Agents commit directly to each repo with proper git history
- **Easy PRs** - Push feature branches to your forks and create PRs to upstream

### Architecture

Uses sibling bare repos with worktrees (not submodules):
- Agents can commit/branch/push independently to each repo
- Same branch name across all three repos for feature work
- No submodule SHA coordination conflicts

### Directory Structure (after `make workspace-init`)

```
/home/ttuser/src_docker/
├── docker_tt/              # This repo (orchestration)
├── docker_tt.git/          # Bare clone for worktrees
├── tt-metal.git/           # Bare clone for worktrees
├── vllm.git/               # Bare clone for worktrees
└── ws/                     # Workspaces
    ├── main/               # Default workspace
    │   ├── docker_tt/      # Worktree (branch: main)
    │   ├── tt-metal/       # Worktree (branch: main)
    │   └── vllm/           # Worktree (branch: dev)
    └── <feature-name>/     # Feature workspaces
        ├── docker_tt/      # Worktree (branch: <feature-name>)
        ├── tt-metal/       # Worktree (branch: <feature-name>)
        └── vllm/           # Worktree (branch: <feature-name>)
```

### Setup

#### 1. Create GitHub Forks

Fork these repositories on GitHub (click "Fork" button on each):
- https://github.com/tenstorrent/tt-metal → your fork
- https://github.com/tenstorrent/vllm → your fork

You'll push feature branches to your forks and create PRs back to upstream.

#### 2. Configure Fork URLs

```bash
cp scripts/workspace.env.example scripts/workspace.env
```

Edit `scripts/workspace.env` with your fork URLs:
```bash
DOCKER_TT_FORK=https://github.com/<your-username>/tt-vllm-simple.git
TT_METAL_FORK=https://github.com/<your-username>/tt-metal.git
VLLM_FORK=https://github.com/<your-username>/vllm.git
```

#### 3. Initialize Workspaces

```bash
make workspace-init
```

This clones all repos as bare repositories and creates the `main` workspace with worktrees.

### Workflow

```bash
# Create a feature workspace
make workspace-create NAME=fix-memory-leak

# Work in the workspace
cd /home/ttuser/src_docker/ws/fix-memory-leak

# Edit code across repos
# - tt-metal/: C++ and Python changes
# - vllm/: Python changes
# - docker_tt/: Build/config changes

# Build and test with your changes
make build-dev WORKSPACE_PATH=$PWD
make run-dev WORKSPACE_PATH=$PWD
make test-dev WORKSPACE_PATH=$PWD

# Commit to each repo independently
cd tt-metal && git add -A && git commit -m "Fix memory leak"
cd ../vllm && git add -A && git commit -m "Fix memory leak"

# Push and create PRs
cd tt-metal && git push origin fix-memory-leak
cd ../vllm && git push origin fix-memory-leak
# Create PRs via GitHub

# Clean up when done
make workspace-delete NAME=fix-memory-leak
```

### Workspace Commands

```bash
make workspace-init         # Clone repos, create main workspace
make workspace-create NAME=x # Create feature workspace with matching branches
make workspace-list         # List all workspaces
make workspace-status NAME=x # Git status across all repos
make workspace-sync         # Fetch from origin and upstream
make workspace-delete NAME=x # Remove workspace and worktrees
```

### Dev Build Commands

```bash
make build-dev              # Build dev image
make run-dev                # Start with main workspace (or set WORKSPACE_PATH)
make test-dev               # Full test cycle
make stop-dev               # Stop dev stack
make logs-dev               # Follow logs
```

### Git Remote Configuration

Each repo has two remotes:
- `origin` - Your fork (for pushing branches, creating PRs)
- `upstream` - Tenstorrent repos (for fetching latest)

Configure in `scripts/workspace.env`.

## Active Porting Efforts

Each model bring-up lives on its own branch (same name across all three repos) with a
`WORKLOG_*.md` file tracking status, run commands, and known issues. Detailed research
and iteration logs live outside git in `/home/ttuser/src_docker/plan/<model_name>/`.

### GLM-4.7-Flash (GLM-4, GLM4 MoE Lite)
1. Create the workspace: `make workspace-create NAME=glm47_flash`
   (This fetches from origin and checks out existing `glm47_flash` branches automatically.)
2. Read `ws/glm47_flash/docker_tt/WORKLOG_GLM47_FLASH.md` for run commands and current status
3. Read `ws/glm47_flash/docker_tt/PLAN_GLM47_FLASH.md` for architecture, history, and next steps
4. The workspace is at `ws/glm47_flash/` -- all three repos have matching branches
5. tt-metal fork: https://github.com/mickg10/tt-metal/tree/glm47_flash
6. vllm fork: https://github.com/mickg10/vllm/tree/glm47_flash

## Team-Based Performance Optimization

For multi-agent performance sprints, use Claude Code's team infrastructure
(`TeamCreate`, `TaskCreate`, `SendMessage`).

### Team Structure

Full doc: `/home/ttuser/src_docker/plan/glm47_flash/team-structure.md`

Three roles only:

| Role | Agent Name | What It Does | What It NEVER Does |
|------|-----------|-------------|-------------------|
| **Team Lead** | `team-lead` | Delegates, coordinates, evaluates results | Edit files, run docker, run benchmarks |
| **Architect** | `architect` | Research via Codex, design optimizations | Edit files, run docker, run benchmarks |
| **Implementer** | `implementer` | Edit code, restart containers, benchmark | Design optimizations, make architectural decisions |

### CRITICAL: Team Lead Delegation Rule

The team lead **MUST NOT** directly:
- Edit any file (`Edit`, `Write` tools)
- Run docker commands (`docker compose`, `docker restart`)
- Run benchmarks (`python tests/bench_decode.py`)
- Modify env files, compose files, or model code

The team lead **ONLY**:
- Creates teams and tasks
- Sends instructions to architect/implementer via `SendMessage`
- Reads files and results to understand state
- Updates `perf-opt.md` with findings (the ONE write exception)

**If you are the team lead and about to edit a file or run docker: STOP.
Send the instruction to the implementer instead.**

### Key Rules

1. **ONE implementer at a time** -- never spawn two (they corrupt code/containers)
2. **Always consult Codex** (`mcp__codex-cli__codex` with model="gpt-5.2") at every architectural decision
3. **Feature-flag everything** -- new optimizations behind env vars with safe defaults
4. **Work in worktrees** -- all changes in `ws/glm47_flash/`, never in main `docker_tt/`
5. **Record everything** -- benchmark results go to `plan/glm47_flash/perf-opt.md`
6. **Architect is long-lived** (keeps context), implementer is ephemeral (per-task)

### The Loop

```
Team Lead ──(design request)──> Architect
Architect ──(analysis)────────> Team Lead
Team Lead ──(task)────────────> Implementer
Implementer ──(results)───────> Team Lead
Team Lead ──(results)─────────> Architect
(repeat)
```

### Quick Launch

```python
TeamCreate(team_name="glm-perf-sprint-N", description="GLM-4.7-Flash perf optimization")
# See plan/glm47_flash/team-structure.md for prompt templates and full details
```

## Upstream Repositories

- tt-metal: https://github.com/tenstorrent/tt-metal (branch: main)
- vLLM fork: https://github.com/tenstorrent/vllm (branch: dev)
