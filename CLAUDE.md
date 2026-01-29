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
├── base/                    # Shared services (WebUI, Monitor)
│   └── docker-compose.yml
├── tt-monitor/              # Device monitoring dashboard
├── tests/                   # Test scripts
└── scripts/                 # Utility scripts
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

## Prebuilt vs Source Build

| Aspect | Prebuilt | Source |
|--------|----------|--------|
| Build time | ~2 min | ~45 min |
| Image size | ~15 GB | ~25 GB |
| Customization | Limited | Full |
| Distributed | No | Yes (Galaxy/TG) |
| Use case | Production | Development |

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

## Upstream Repositories

- tt-metal: https://github.com/tenstorrent/tt-metal (branch: main)
- vLLM fork: https://github.com/tenstorrent/vllm (branch: dev)
