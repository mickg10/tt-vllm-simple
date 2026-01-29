# Plan: Restructure Docker Setup with Prebuilt and Source Build Variants

## Overview

Restructure the docker_tt project to have two independent directories (`prebuilt/` and `from_source/`), each with its own docker-compose.yml that includes shared base services. Add comprehensive testing and a top-level Makefile.

---

## Final Directory Structure

```
docker_tt/
├── Makefile                    # Top-level build/test/run commands
├── CLAUDE.md                   # Updated project documentation
├── entrypoint.sh               # Shared entrypoint for vLLM
├── base/
│   └── docker-compose.yml      # Shared services: open-webui + tt-monitor
├── prebuilt/
│   ├── docker-compose.yml      # Includes base, adds vllm-tt (prebuilt)
│   ├── Dockerfile              # Uses ghcr.io base image
│   └── .env.example            # Default env vars for prebuilt
├── from_source/
│   ├── docker-compose.yml      # Includes base, adds vllm-tt (source)
│   ├── Dockerfile              # Full source build with distributed
│   └── .env.example            # Default env vars for source build
├── tt-monitor/
│   ├── Dockerfile
│   └── ... (existing files)
├── tests/
│   ├── test_health.sh          # Health check tests
│   ├── test_models.sh          # Model listing tests
│   ├── test_completion.sh      # Completion API tests
│   └── run_all.sh              # Run all tests
└── scripts/
    └── wait_for_healthy.sh     # Wait for vLLM to be ready
```

---

## Task Breakdown

### Phase 1: Create Base Infrastructure

#### Task 1.1: Create base/docker-compose.yml
**File:** `base/docker-compose.yml`
**Description:** Shared services (open-webui, tt-monitor) with Docker bridge networking.

```yaml
networks:
  tt-net:
    driver: bridge

services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    networks:
      - tt-net
    ports:
      - "3000:3000"
    environment:
      - OPENAI_API_KEY=not-needed
      - OPENAI_API_BASE_URL=http://vllm-tt:8088/v1
      - OLLAMA_API_BASE_URL=
      - ENV=prod
      - PORT=3000
      - USE_OLLAMA_DOCKER=false
      - SCARF_NO_ANALYTICS=true
      - DO_NOT_TRACK=true
      - ANONYMIZED_TELEMETRY=false
      - RAG_EMBEDDING_MODEL=
      - RAG_EMBEDDING_ENGINE=
      - USE_EMBEDDING_MODEL_DOCKER=
      - ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION=False
      - ENABLE_RAG_WEB_SEARCH=False
      - HF_HUB_OFFLINE=1
    volumes:
      - open-webui-data:/app/backend/data
    restart: unless-stopped
    depends_on:
      vllm-tt:
        condition: service_healthy

  tt-monitor:
    build:
      context: ../tt-monitor
    image: tt-monitor:latest
    networks:
      - tt-net
    ports:
      - "9090:9090"
    volumes:
      - /sys:/sys:ro
      - tt-monitor-data:/data
    environment:
      - TT_MONITOR_MOCK=${TT_MONITOR_MOCK:-0}
      - FRONTEND_PATH=/app/frontend/dist
      - VLLM_METRICS_URL=http://vllm-tt:8088/metrics
      - TT_MONITOR_DATA_DIR=/data
    restart: unless-stopped

volumes:
  open-webui-data:
  tt-monitor-data:
```

#### Task 1.2: Create shared entrypoint.sh
**File:** `entrypoint.sh` (root level, shared)
**Description:** Entrypoint script that both Dockerfiles will use.

```bash
#!/bin/bash
set -e

# Source tt-metal environment
export TT_METAL_HOME=${TT_METAL_HOME:-/tt-metal}
export PYTHONPATH="${TT_METAL_HOME}:${PYTHONPATH}"

# Activate virtual environment if it exists
if [ -f "/opt/venv/bin/activate" ]; then
    source /opt/venv/bin/activate
fi

# Register TT models with vLLM
python -c "import vllm.model_executor.models.tt_llama; import vllm.model_executor.models.tt_qwen3" 2>/dev/null || true

# Default model
MODEL=${HF_MODEL:-Qwen/Qwen3-0.6B}

echo "Starting vLLM server with args: --model $MODEL $@"

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --disable-frontend-multiprocessing \
    "$@"
```

#### Task 1.3: Create scripts/wait_for_healthy.sh
**File:** `scripts/wait_for_healthy.sh`
**Description:** Wait for vLLM to become healthy (for tests).

```bash
#!/bin/bash
# Wait for vLLM to be healthy
# Usage: ./wait_for_healthy.sh [timeout_seconds] [port]

TIMEOUT=${1:-300}
PORT=${2:-8088}
INTERVAL=5

echo "Waiting for vLLM to be healthy on port $PORT (timeout: ${TIMEOUT}s)..."

elapsed=0
while [ $elapsed -lt $TIMEOUT ]; do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "vLLM is healthy after ${elapsed}s"
        exit 0
    fi
    sleep $INTERVAL
    elapsed=$((elapsed + INTERVAL))
    echo "  Waiting... (${elapsed}s elapsed)"
done

echo "ERROR: vLLM did not become healthy within ${TIMEOUT}s"
exit 1
```

---

### Phase 2: Create Prebuilt Variant

#### Task 2.1: Create prebuilt/Dockerfile
**File:** `prebuilt/Dockerfile`
**Description:** Fast build using pre-built base image from ghcr.io.

```dockerfile
# syntax=docker/dockerfile:1
# Prebuilt variant - uses pre-compiled tt-metal from ghcr.io
FROM ghcr.io/tenstorrent/tt-inference-server:v0.5.0 AS base

# Install any additional dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy entrypoint
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8088/health || exit 1

WORKDIR /app
EXPOSE 8088

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--max-model-len", "32768", "--block-size", "64", "--max-num-seqs", "32", "--port", "8088"]
```

#### Task 2.2: Create prebuilt/docker-compose.yml
**File:** `prebuilt/docker-compose.yml`
**Description:** Compose file that includes base and adds prebuilt vllm-tt service.

```yaml
include:
  - path: ../base/docker-compose.yml

services:
  vllm-tt:
    build:
      context: ..
      dockerfile: prebuilt/Dockerfile
    image: vllm-tt:latest
    networks:
      - tt-net
    ports:
      - "8088:8088"
    devices:
      - /dev/tenstorrent:/dev/tenstorrent
    volumes:
      - /dev/hugepages-1G:/dev/hugepages-1G
      - ${HOME}/.cache/huggingface:/cache/huggingface
    environment:
      - HF_MODEL=${HF_MODEL:-Qwen/Qwen3-0.6B}
      - HF_TOKEN=${HF_TOKEN:-}
      - HF_HOME=/cache/huggingface
      - VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-300000}
      - MESH_DEVICE=${MESH_DEVICE:-N300}
      - WH_ARCH_YAML=${WH_ARCH_YAML:-}
    command:
      - --max-model-len=${MAX_MODEL_LEN:-32768}
      - --max-num-seqs=${MAX_NUM_SEQS:-32}
      - --block-size=64
      - --port=8088
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8088/health"]
      interval: 30s
      timeout: 10s
      start_period: 180s
      retries: 3
    ipc: host
    privileged: true
    restart: unless-stopped
```

#### Task 2.3: Create prebuilt/.env.example
**File:** `prebuilt/.env.example`
**Description:** Default environment variables for prebuilt variant.

```bash
# Prebuilt variant defaults
# Copy to .env and customize as needed

# Model configuration
HF_MODEL=Qwen/Qwen3-0.6B
HF_TOKEN=

# Device configuration
# Options: N150, N300, T3K, TG (Galaxy)
MESH_DEVICE=N300
WH_ARCH_YAML=

# vLLM settings
VLLM_RPC_TIMEOUT=300000
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=32

# Monitoring (set to 1 for mock mode without hardware)
TT_MONITOR_MOCK=0
```

---

### Phase 3: Create Source Build Variant

#### Task 3.1: Create from_source/Dockerfile
**File:** `from_source/Dockerfile`
**Description:** Full source build with distributed support, parallel compilation.

```dockerfile
# syntax=docker/dockerfile:1
# Source build variant - compiles tt-metal and vLLM from source
FROM ubuntu:22.04 AS base

# Build args
ARG TT_METAL_REF=main
ARG VLLM_REF=dev
ARG DEBIAN_FRONTEND=noninteractive
ARG MAKEFLAGS="-j16"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build pkg-config \
    wget curl software-properties-common gnupg ca-certificates \
    python3 python3-dev python3-pip python3-venv \
    libhwloc-dev libnuma-dev libatomic1 libtbb-dev \
    libcapstone-dev xxd libssl-dev \
    # For distributed support
    libopenmpi-dev openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Install clang-20 (required compiler)
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-20 main" > /etc/apt/sources.list.d/llvm.list \
    && apt-get update && apt-get install -y --no-install-recommends clang-20 lld-20 \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Clone tt-metal (shallow clone for speed)
WORKDIR /
RUN git config --global http.postBuffer 524288000 \
    && git clone --depth 1 -b ${TT_METAL_REF} https://github.com/tenstorrent/tt-metal.git \
    && cd tt-metal \
    && git submodule update --init --recursive --depth 1

WORKDIR /tt-metal

# Install dependencies (WITH distributed support for Galaxy)
RUN ./install_dependencies.sh --docker

# Build tt-metal C++ components with parallel compilation
# Using MAKEFLAGS for parallelism in underlying make calls
ENV MAKEFLAGS=${MAKEFLAGS}
RUN CMAKE_BUILD_PARALLEL_LEVEL=16 ./build_metal.sh

# Create Python virtual environment and install tt-metal
ENV PYTHON_ENV_DIR=/opt/venv
RUN ./create_venv.sh --env-dir /opt/venv

# Clone vLLM
WORKDIR /
RUN git clone --depth 1 -b ${VLLM_REF} https://github.com/tenstorrent/vllm.git

# Install vLLM for TT target
WORKDIR /vllm
ENV VLLM_TARGET_DEVICE=tt
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV=/opt/venv

RUN . /opt/venv/bin/activate \
    && uv pip install -e . \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       --index-strategy unsafe-best-match

# Runtime configuration
ENV TT_METAL_HOME=/tt-metal
ENV PYTHONPATH=/tt-metal

# Copy entrypoint
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -sf http://localhost:8088/health || exit 1

WORKDIR /vllm
EXPOSE 8088

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--max-model-len", "32768", "--block-size", "64", "--max-num-seqs", "32", "--port", "8088"]
```

#### Task 3.2: Create from_source/docker-compose.yml
**File:** `from_source/docker-compose.yml`
**Description:** Compose file that includes base and adds source-built vllm-tt service.

```yaml
include:
  - path: ../base/docker-compose.yml

services:
  vllm-tt:
    build:
      context: ..
      dockerfile: from_source/Dockerfile
      args:
        TT_METAL_REF: ${TT_METAL_REF:-main}
        VLLM_REF: ${VLLM_REF:-dev}
        MAKEFLAGS: "-j${BUILD_JOBS:-16}"
    image: vllm-tt-source:latest
    networks:
      - tt-net
    ports:
      - "8088:8088"
    devices:
      - /dev/tenstorrent:/dev/tenstorrent
    volumes:
      - /dev/hugepages-1G:/dev/hugepages-1G
      - ${HOME}/.cache/huggingface:/cache/huggingface
    environment:
      - HF_MODEL=${HF_MODEL:-Qwen/Qwen3-0.6B}
      - HF_TOKEN=${HF_TOKEN:-}
      - HF_HOME=/cache/huggingface
      - VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-600000}
      - MESH_DEVICE=${MESH_DEVICE:-N300}
      - WH_ARCH_YAML=${WH_ARCH_YAML:-}
      - TT_METAL_NUM_COMMAND_QUEUES=${TT_METAL_NUM_COMMAND_QUEUES:-}
    command:
      - --max-model-len=${MAX_MODEL_LEN:-32768}
      - --max-num-seqs=${MAX_NUM_SEQS:-32}
      - --block-size=64
      - --port=8088
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8088/health"]
      interval: 30s
      timeout: 10s
      start_period: 300s
      retries: 3
    ipc: host
    privileged: true
    restart: unless-stopped
```

#### Task 3.3: Create from_source/.env.example
**File:** `from_source/.env.example`
**Description:** Default environment variables for source build variant.

```bash
# Source build variant defaults
# Copy to .env and customize as needed

# Build configuration
TT_METAL_REF=main
VLLM_REF=dev
BUILD_JOBS=16

# Model configuration
HF_MODEL=Qwen/Qwen3-0.6B
HF_TOKEN=

# Device configuration
# Options: N150, N300, T3K, TG (Galaxy)
MESH_DEVICE=N300
WH_ARCH_YAML=

# vLLM settings (longer timeout for source builds)
VLLM_RPC_TIMEOUT=600000
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=32

# Advanced tt-metal settings
TT_METAL_NUM_COMMAND_QUEUES=

# Monitoring (set to 1 for mock mode without hardware)
TT_MONITOR_MOCK=0

# Debug options (uncomment to enable)
#VLLM_LOGGING_LEVEL=DEBUG
#TT_METAL_LOGGER_LEVEL=DEBUG
```

---

### Phase 4: Create Test Suite

#### Task 4.1: Create tests/test_health.sh
**File:** `tests/test_health.sh`
**Description:** Test that vLLM health endpoint responds.

```bash
#!/bin/bash
set -e

PORT=${1:-8088}
echo "Testing health endpoint on port $PORT..."

response=$(curl -sf "http://localhost:${PORT}/health" 2>&1) || {
    echo "FAIL: Health check failed"
    exit 1
}

echo "PASS: Health check succeeded"
exit 0
```

#### Task 4.2: Create tests/test_models.sh
**File:** `tests/test_models.sh`
**Description:** Test that model listing works and returns expected model.

```bash
#!/bin/bash
set -e

PORT=${1:-8088}
EXPECTED_MODEL=${2:-Qwen/Qwen3-0.6B}

echo "Testing model listing on port $PORT..."
echo "Expected model: $EXPECTED_MODEL"

response=$(curl -sf "http://localhost:${PORT}/v1/models") || {
    echo "FAIL: Could not fetch models"
    exit 1
}

# Check if expected model is in response
if echo "$response" | grep -q "$EXPECTED_MODEL"; then
    echo "PASS: Model '$EXPECTED_MODEL' found"
    echo "Response: $response" | python3 -m json.tool 2>/dev/null || echo "$response"
    exit 0
else
    echo "FAIL: Model '$EXPECTED_MODEL' not found in response"
    echo "Response: $response"
    exit 1
fi
```

#### Task 4.3: Create tests/test_completion.sh
**File:** `tests/test_completion.sh`
**Description:** Test that chat completion API works.

```bash
#!/bin/bash
set -e

PORT=${1:-8088}
MODEL=${2:-Qwen/Qwen3-0.6B}

echo "Testing chat completion on port $PORT with model $MODEL..."

response=$(curl -sf "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in exactly 3 words.\"}],
        \"max_tokens\": 20
    }") || {
    echo "FAIL: Chat completion request failed"
    exit 1
}

# Check for choices in response
if echo "$response" | grep -q '"choices"'; then
    echo "PASS: Chat completion succeeded"
    echo "Response:"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    exit 0
else
    echo "FAIL: Invalid response (no choices)"
    echo "Response: $response"
    exit 1
fi
```

#### Task 4.4: Create tests/run_all.sh
**File:** `tests/run_all.sh`
**Description:** Run all tests in sequence.

```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=${1:-8088}
MODEL=${2:-Qwen/Qwen3-0.6B}

echo "========================================"
echo "Running all tests"
echo "Port: $PORT"
echo "Model: $MODEL"
echo "========================================"

FAILED=0

echo ""
echo "--- Test 1: Health Check ---"
if "$SCRIPT_DIR/test_health.sh" "$PORT"; then
    echo "✓ Health check passed"
else
    echo "✗ Health check failed"
    FAILED=1
fi

echo ""
echo "--- Test 2: Model Listing ---"
if "$SCRIPT_DIR/test_models.sh" "$PORT" "$MODEL"; then
    echo "✓ Model listing passed"
else
    echo "✗ Model listing failed"
    FAILED=1
fi

echo ""
echo "--- Test 3: Chat Completion ---"
if "$SCRIPT_DIR/test_completion.sh" "$PORT" "$MODEL"; then
    echo "✓ Chat completion passed"
else
    echo "✗ Chat completion failed"
    FAILED=1
fi

echo ""
echo "========================================"
if [ $FAILED -eq 0 ]; then
    echo "All tests PASSED"
    exit 0
else
    echo "Some tests FAILED"
    exit 1
fi
```

---

### Phase 5: Create Top-Level Makefile

#### Task 5.1: Create Makefile
**File:** `Makefile`
**Description:** Top-level Makefile with all build/test/run targets. NO default target that runs just one thing.

```makefile
# TT vLLM Docker Setup Makefile
# Usage: make <target>
# Run 'make help' for available targets

SHELL := /bin/bash
.PHONY: help build-prebuilt build-source test-prebuilt test-source \
        run-prebuilt run-source stop-prebuilt stop-source \
        logs-prebuilt logs-source clean reset-devices

# No default target - must specify explicitly
.DEFAULT_GOAL := help

# Configuration
PORT ?= 8088
MODEL ?= Qwen/Qwen3-0.6B
WAIT_TIMEOUT ?= 300
BUILD_JOBS ?= 16

#==============================================================================
# Help
#==============================================================================

help:
	@echo "TT vLLM Docker Setup"
	@echo ""
	@echo "Build targets:"
	@echo "  make build-prebuilt     Build prebuilt variant (fast)"
	@echo "  make build-source       Build from-source variant (slow, ~45min)"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-prebuilt       Start prebuilt stack (vLLM + WebUI + Monitor)"
	@echo "  make run-source         Start source-built stack"
	@echo ""
	@echo "Test targets:"
	@echo "  make test-prebuilt      Test prebuilt variant (builds, starts, tests, stops)"
	@echo "  make test-source        Test source variant (builds, starts, tests, stops)"
	@echo ""
	@echo "Stop targets:"
	@echo "  make stop-prebuilt      Stop prebuilt stack"
	@echo "  make stop-source        Stop source stack"
	@echo ""
	@echo "Utility targets:"
	@echo "  make logs-prebuilt      Follow prebuilt logs"
	@echo "  make logs-source        Follow source logs"
	@echo "  make reset-devices      Reset TT devices (run before starting)"
	@echo "  make clean              Remove all containers and images"
	@echo ""
	@echo "Configuration (override with VAR=value):"
	@echo "  PORT=$(PORT)            vLLM API port"
	@echo "  MODEL=$(MODEL)          Model to serve"
	@echo "  WAIT_TIMEOUT=$(WAIT_TIMEOUT)       Seconds to wait for healthy"
	@echo "  BUILD_JOBS=$(BUILD_JOBS)          Parallel build jobs"

#==============================================================================
# Build Targets
#==============================================================================

build-prebuilt:
	@echo "Building prebuilt variant..."
	cd prebuilt && docker compose build

build-source:
	@echo "Building source variant with $(BUILD_JOBS) parallel jobs..."
	cd from_source && BUILD_JOBS=$(BUILD_JOBS) docker compose build

#==============================================================================
# Run Targets
#==============================================================================

run-prebuilt:
	@echo "Starting prebuilt stack..."
	@if [ ! -f prebuilt/.env ]; then \
		echo "Creating prebuilt/.env from .env.example..."; \
		cp prebuilt/.env.example prebuilt/.env; \
	fi
	cd prebuilt && docker compose up -d
	@echo ""
	@echo "Stack starting. Services will be available at:"
	@echo "  vLLM API:    http://localhost:$(PORT)/v1"
	@echo "  Open WebUI:  http://localhost:3000"
	@echo "  TT Monitor:  http://localhost:9090"
	@echo ""
	@echo "Run 'make logs-prebuilt' to follow logs"

run-source:
	@echo "Starting source-built stack..."
	@if [ ! -f from_source/.env ]; then \
		echo "Creating from_source/.env from .env.example..."; \
		cp from_source/.env.example from_source/.env; \
	fi
	cd from_source && docker compose up -d
	@echo ""
	@echo "Stack starting. Services will be available at:"
	@echo "  vLLM API:    http://localhost:$(PORT)/v1"
	@echo "  Open WebUI:  http://localhost:3000"
	@echo "  TT Monitor:  http://localhost:9090"
	@echo ""
	@echo "Run 'make logs-source' to follow logs"

#==============================================================================
# Stop Targets
#==============================================================================

stop-prebuilt:
	@echo "Stopping prebuilt stack..."
	cd prebuilt && docker compose down

stop-source:
	@echo "Stopping source stack..."
	cd from_source && docker compose down

#==============================================================================
# Test Targets
#==============================================================================

test-prebuilt: build-prebuilt
	@echo "============================================"
	@echo "Testing prebuilt variant"
	@echo "============================================"
	@# Ensure stopped first
	cd prebuilt && docker compose down 2>/dev/null || true
	@# Copy env if needed
	@if [ ! -f prebuilt/.env ]; then cp prebuilt/.env.example prebuilt/.env; fi
	@# Start stack
	cd prebuilt && docker compose up -d
	@# Wait for healthy
	@echo "Waiting for vLLM to be healthy (timeout: $(WAIT_TIMEOUT)s)..."
	@./scripts/wait_for_healthy.sh $(WAIT_TIMEOUT) $(PORT)
	@# Run tests
	@echo ""
	@./tests/run_all.sh $(PORT) "$(MODEL)"
	@# Stop stack
	@echo ""
	@echo "Stopping stack..."
	cd prebuilt && docker compose down
	@echo ""
	@echo "============================================"
	@echo "Prebuilt tests complete"
	@echo "============================================"

test-source: build-source
	@echo "============================================"
	@echo "Testing source variant"
	@echo "============================================"
	@# Ensure stopped first
	cd from_source && docker compose down 2>/dev/null || true
	@# Copy env if needed
	@if [ ! -f from_source/.env ]; then cp from_source/.env.example from_source/.env; fi
	@# Start stack
	cd from_source && docker compose up -d
	@# Wait for healthy (longer timeout for source builds)
	@echo "Waiting for vLLM to be healthy (timeout: $(WAIT_TIMEOUT)s)..."
	@./scripts/wait_for_healthy.sh $(WAIT_TIMEOUT) $(PORT)
	@# Run tests
	@echo ""
	@./tests/run_all.sh $(PORT) "$(MODEL)"
	@# Stop stack
	@echo ""
	@echo "Stopping stack..."
	cd from_source && docker compose down
	@echo ""
	@echo "============================================"
	@echo "Source tests complete"
	@echo "============================================"

#==============================================================================
# Utility Targets
#==============================================================================

logs-prebuilt:
	cd prebuilt && docker compose logs -f

logs-source:
	cd from_source && docker compose logs -f

reset-devices:
	@echo "Resetting TT devices..."
	tt-smi -r 0,1,2,3 || echo "Warning: Device reset may have partially failed"
	@echo "Waiting for devices to reinitialize..."
	sleep 5
	@echo "Device reset complete"

clean:
	@echo "Stopping all containers..."
	cd prebuilt && docker compose down --rmi local 2>/dev/null || true
	cd from_source && docker compose down --rmi local 2>/dev/null || true
	@echo "Removing images..."
	docker rmi vllm-tt:latest vllm-tt-source:latest tt-monitor:latest 2>/dev/null || true
	@echo "Clean complete"
```

---

### Phase 6: Update Documentation

#### Task 6.1: Update CLAUDE.md
**File:** `CLAUDE.md`
**Description:** Updated project documentation reflecting new structure.

```markdown
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
```

---

### Phase 7: Cleanup Old Files

#### Task 7.1: Remove old files
**Files to remove:**
- `docker_tt/Dockerfile` (replaced by prebuilt/Dockerfile)
- `docker_tt/docker-compose.yml` (replaced by prebuilt/ and from_source/)
- `docker_tt/fullbuild/` (entire directory, replaced by from_source/)

---

## Task Summary

| Phase | Task | File | Description |
|-------|------|------|-------------|
| 1 | 1.1 | base/docker-compose.yml | Shared services with bridge network |
| 1 | 1.2 | entrypoint.sh | Shared entrypoint script |
| 1 | 1.3 | scripts/wait_for_healthy.sh | Health check wait script |
| 2 | 2.1 | prebuilt/Dockerfile | Fast prebuilt image |
| 2 | 2.2 | prebuilt/docker-compose.yml | Prebuilt compose with include |
| 2 | 2.3 | prebuilt/.env.example | Prebuilt defaults |
| 3 | 3.1 | from_source/Dockerfile | Full source build with -j16 |
| 3 | 3.2 | from_source/docker-compose.yml | Source compose with include |
| 3 | 3.3 | from_source/.env.example | Source build defaults |
| 4 | 4.1 | tests/test_health.sh | Health endpoint test |
| 4 | 4.2 | tests/test_models.sh | Model listing test |
| 4 | 4.3 | tests/test_completion.sh | Completion API test |
| 4 | 4.4 | tests/run_all.sh | Test runner |
| 5 | 5.1 | Makefile | Top-level make targets |
| 6 | 6.1 | CLAUDE.md | Updated documentation |
| 7 | 7.1 | (cleanup) | Remove old files |

---

## Verification Steps

After implementation:

1. **Test prebuilt variant:**
   ```bash
   make reset-devices
   make test-prebuilt
   ```

2. **Test source variant:**
   ```bash
   make reset-devices
   make test-source
   ```

3. **Verify services accessible:**
   - http://localhost:8088/v1/models
   - http://localhost:3000
   - http://localhost:9090

4. **Verify parallel build:**
   - Source build should show `-j16` in cmake/make output
   - Build logs should show parallel compilation

---

## Notes

- Source build includes distributed support for Galaxy (TG)
- Both variants use Docker bridge networking (`tt-net`)
- Services communicate via service names (e.g., `vllm-tt:8088`)
- Health checks ensure proper startup sequencing
- Tests are automated and run in CI-friendly manner
