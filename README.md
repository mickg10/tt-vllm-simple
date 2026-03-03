# TT vLLM Docker Setup

Docker setup for running vLLM with Tenstorrent hardware acceleration.

## Quick Start

```bash
# Using prebuilt image (recommended)
make build-prebuilt
make run-prebuilt

# Access services
open http://localhost:3000    # Open WebUI
open http://localhost:8088/v1 # vLLM API
open http://localhost:9090    # TT Monitor
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| vLLM API | 8088 | OpenAI-compatible API |
| Open WebUI | 3000 | Chat interface |
| TT Monitor | 9090 | Device monitoring dashboard |

## Build Variants

| Variant | Build Time | Use Case |
|---------|------------|----------|
| `prebuilt` | ~2 min | Production, quick start |
| `from_source` | ~45 min | CI/Testing, full customization |
| `dev` | ~5 min | Active development with live code |

```bash
make build-prebuilt   # Fast, uses pre-built image
make build-source     # Compiles tt-metal and vllm from source
make build-dev        # For development with bind-mounted sources
```

## Configuration

Copy and edit the environment file for your variant:

```bash
cd prebuilt  # or from_source, dev
cp .env.example .env
```

Key settings:
- `HF_MODEL` - Model to serve (default: Qwen/Qwen3-0.6B)
- `HF_TOKEN` - HuggingFace token for gated models
- `MESH_DEVICE` - Device config: N150, N300, T3K, TG

## Agentic Development Workflow

This repo supports coordinated development across three repositories using git worktrees - ideal for AI agents (Claude, Codex) or parallel feature development.

### Why Worktrees?

| Traditional Approach | Worktree Approach |
|---------------------|-------------------|
| Single checkout, branch switching | Multiple parallel workspaces |
| Submodule SHA coordination | Independent repo commits |
| One task at a time | Parallel feature development |
| Conflicts when switching | Isolated workspaces |

### Setup

#### 1. Fork the Repositories

Fork these on GitHub:
- https://github.com/tenstorrent/tt-metal
- https://github.com/tenstorrent/vllm

#### 2. Configure Your Forks

```bash
cp scripts/workspace.env.example scripts/workspace.env
# Edit with your fork URLs
```

#### 3. Initialize

```bash
make workspace-init
```

This creates:
```
/home/ttuser/src_docker/
├── docker_tt/          # This repo (working copy)
├── docker_tt.git/      # Bare clone for worktrees
├── tt-metal.git/       # Bare clone for worktrees
├── vllm.git/           # Bare clone for worktrees
└── ws/
    └── main/           # Default workspace
        ├── docker_tt/  # Worktree (branch: main)
        ├── tt-metal/   # Worktree (branch: main)
        └── vllm/       # Worktree (branch: dev)
```

### Workflow

```bash
# Create feature workspace
make workspace-create NAME=fix-memory-leak
cd /home/ttuser/src_docker/ws/fix-memory-leak

# Make changes across repos
vim tt-metal/...
vim vllm/...

# Build and test
make build-dev WORKSPACE_PATH=$PWD
make run-dev WORKSPACE_PATH=$PWD
make test-dev WORKSPACE_PATH=$PWD

# Commit and push (same branch name in each repo)
cd tt-metal && git add -A && git commit -m "Fix memory leak"
cd ../vllm && git add -A && git commit -m "Fix memory leak"

# Push both repos
cd ../tt-metal && git push origin fix-memory-leak
cd ../vllm && git push origin fix-memory-leak

# Clean up when done
make workspace-delete NAME=fix-memory-leak
```

### Workspace Commands

```bash
make workspace-init          # First-time setup
make workspace-create NAME=x # New feature workspace
make workspace-list          # Show all workspaces
make workspace-status NAME=x # Git status across repos
make workspace-sync          # Fetch from upstream
make workspace-delete NAME=x # Remove workspace
```

## Troubleshooting

### Device initialization failed
```bash
make reset-devices
make run-prebuilt
```

### Model download issues
```bash
echo "HF_TOKEN=hf_xxxx" >> prebuilt/.env
```

### Port already in use
```bash
make stop-prebuilt
make stop-source
make stop-dev
```

## Supported Models

Models with TT backend support:
- Qwen3 (0.6B, 4B, 8B, 14B, 32B)
- Qwen2/2.5
- Llama 3.x
- Mistral
- Gemma 3
- DeepSeek V3

See [tt-metal model support](https://github.com/tenstorrent/tt-metal/tree/main/models) for the full list.

## Galaxy Wormhole: GLM-4.7-Flash

GLM-4.7-Flash running on a 32-chip Galaxy Wormhole TG mesh (TP=8 x DP=4).

### Best Results

| Metric | Value |
|--------|-------|
| bs=1 throughput | 8.1 tok/s (115.7ms ITL) |
| bs=32 aggregate throughput | 250.2 tok/s (118.2ms ITL) |
| vs T3K (bs=1) | **+15.7%** |

### Known Issues

- Trace replay hang on 32-chip mesh due to device barrier race condition (requires TT firmware fix). See [trace-replay-hang-analysis.md](docs/glm_47_flash/galaxy_wormhole/trace-replay-hang-analysis.md).

### Documentation

- [Performance optimization log](docs/glm_47_flash/galaxy_wormhole/perf-opt.md)
- [Trace replay hang analysis](docs/glm_47_flash/galaxy_wormhole/trace-replay-hang-analysis.md)
- [Galaxy system state](docs/glm_47_flash/galaxy_wormhole/galaxy-system-state.md)

## Upstream Repositories

- tt-metal: https://github.com/tenstorrent/tt-metal
- vLLM fork: https://github.com/tenstorrent/vllm
