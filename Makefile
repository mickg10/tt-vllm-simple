# TT vLLM Docker Setup Makefile
# Usage: make <target>
# Run 'make help' for available targets

SHELL := /bin/bash
.PHONY: help build-prebuilt build-source build-dev test-prebuilt test-source test-dev \
        run-prebuilt run-source run-dev stop-prebuilt stop-source stop-dev \
        build-device run-device stop-device logs-device wait-device verify-device \
        diagnostics-device diagnostics-galaxy shell-device shell-galaxy \
        run-galaxy stop-galaxy logs-galaxy wait-galaxy verify-galaxy \
        logs-prebuilt logs-source logs-dev clean reset-devices \
        workspace-init workspace-create workspace-list workspace-delete workspace-status workspace-sync

# No default target - must specify explicitly
.DEFAULT_GOAL := help

# Configuration
PORT ?= 8088
MODEL ?= Qwen/Qwen3-0.6B
WAIT_TIMEOUT ?= 300
BUILD_JOBS ?= 16
WORKSPACE_PATH ?= $(shell ./scripts/workspace.sh path main 2>/dev/null || echo "")

# Device configuration (override for different targets: galaxy, t3k, blackhole, etc.)
# Usage: make run-device DEVICE_ENV=dev/.env.glm47.galaxy DEVICE_COMPOSE_EXTRA="-f dev/docker-compose.galaxy.yml"
DEVICE_ENV ?= dev/.env.glm47.galaxy
DEVICE_COMPOSE_EXTRA ?= -f dev/docker-compose.galaxy.yml
DEVICE_PROJECT ?= glm-flash
DEVICE_CONTAINER = $(DEVICE_PROJECT)-vllm-tt-1
DEVICE_WAIT_TIMEOUT ?= 5400

#==============================================================================
# Help
#==============================================================================

help:
	@echo "TT vLLM Docker Setup"
	@echo ""
	@echo "Build targets:"
	@echo "  make build-prebuilt     Build prebuilt variant (fast)"
	@echo "  make build-source       Build from-source variant (slow)"
	@echo "  make build-dev          Build dev variant (uses bind mounts)"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-prebuilt       Start prebuilt stack (vLLM + WebUI + Monitor)"
	@echo "  make run-source         Start source-built stack"
	@echo "  make run-dev            Start dev stack with workspace sources"
	@echo ""
	@echo "Test targets:"
	@echo "  make test-prebuilt      Test prebuilt variant (builds, starts, tests, stops)"
	@echo "  make test-source        Test source variant (builds, starts, tests, stops)"
	@echo "  make test-dev           Test dev variant with workspace sources"
	@echo ""
	@echo "Stop targets:"
	@echo "  make stop-prebuilt      Stop prebuilt stack"
	@echo "  make stop-source        Stop source stack"
	@echo "  make stop-dev           Stop dev stack"
	@echo ""
	@echo "Device targets (Galaxy, T3K, Blackhole, etc.):"
	@echo "  make build-device       Build dev image for target device"
	@echo "  make run-device         Start model on target device"
	@echo "  make stop-device        Stop device container"
	@echo "  make logs-device        Follow device logs"
	@echo "  make wait-device        Wait for container to become healthy"
	@echo "  make verify-device      Verify coherent output"
	@echo "  make diagnostics-device Run tt-metal diagnostics (eth status, NOC, triage)"
	@echo "  make shell-device       Interactive bash with all tt-tools (no vLLM)"
	@echo ""
	@echo "One-command scripts (clone + run):"
	@echo "  ./scripts/deploy-galaxy.sh                    # clone + deploy vLLM"
	@echo "  ./scripts/deploy-galaxy.sh --mode diagnostics # clone + diagnostics only"
	@echo "  ./scripts/deploy-galaxy.sh --mode bash        # clone + interactive shell"
	@echo ""
	@echo "  Shortcuts:"
	@echo "  make run-galaxy         = run-device with Galaxy Wormhole config"
	@echo "  make run-t3k            = run-device with T3K config (TODO)"
	@echo ""
	@echo "Workspace targets (agentic development):"
	@echo "  make workspace-init     Clone repos as bare, create main workspace"
	@echo "  make workspace-create   Create feature workspace (NAME=<name>)"
	@echo "  make workspace-list     List all workspaces"
	@echo "  make workspace-delete   Delete workspace (NAME=<name>)"
	@echo "  make workspace-status   Show status (NAME=<name>, default: main)"
	@echo "  make workspace-sync     Fetch from origin and upstream"
	@echo ""
	@echo "Utility targets:"
	@echo "  make logs-prebuilt      Follow prebuilt logs"
	@echo "  make logs-source        Follow source logs"
	@echo "  make logs-dev           Follow dev logs"
	@echo "  make reset-devices      Reset TT devices (run before starting)"
	@echo "  make clean              Remove all containers and images"
	@echo ""
	@echo "Configuration (override with VAR=value):"
	@echo "  PORT=$(PORT)            vLLM API port"
	@echo "  MODEL=$(MODEL)          Model to serve"
	@echo "  WAIT_TIMEOUT=$(WAIT_TIMEOUT)       Seconds to wait for healthy"
	@echo "  BUILD_JOBS=$(BUILD_JOBS)          Parallel build jobs"
	@echo "  WORKSPACE_PATH=<path>   Workspace path for dev builds"
	@echo "  DEVICE_ENV=$(DEVICE_ENV)  Device env file"
	@echo "  DEVICE_COMPOSE_EXTRA=$(DEVICE_COMPOSE_EXTRA)  Extra compose file"
	@echo "  DEVICE_PROJECT=$(DEVICE_PROJECT)  Compose project name"
	@echo "  DEVICE_WAIT_TIMEOUT=$(DEVICE_WAIT_TIMEOUT)  Health wait (seconds)"

#==============================================================================
# Build Targets
#==============================================================================

build-prebuilt:
	@echo "Building prebuilt variant..."
	cd prebuilt && docker compose build

build-source:
	@echo "Building source variant with $(BUILD_JOBS) parallel jobs..."
	cd from_source && BUILD_JOBS=$(BUILD_JOBS) docker compose build

build-dev:
	@echo "Building dev variant..."
	cd dev && docker compose build

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

run-dev:
	@echo "Starting dev stack..."
	@if [ ! -f dev/.env ]; then \
		echo "Creating dev/.env from .env.example..."; \
		cp dev/.env.example dev/.env; \
	fi
	@if [ -z "$(WORKSPACE_PATH)" ]; then \
		echo "ERROR: WORKSPACE_PATH not set. Run 'make workspace-init' first or set WORKSPACE_PATH."; \
		exit 1; \
	fi
	@if [ ! -d "$(WORKSPACE_PATH)/tt-metal" ]; then \
		echo "ERROR: tt-metal not found at $(WORKSPACE_PATH)/tt-metal"; \
		echo "Make sure WORKSPACE_PATH points to a valid workspace."; \
		exit 1; \
	fi
	cd dev && WORKSPACE_PATH=$(WORKSPACE_PATH) docker compose up -d
	@echo ""
	@echo "Dev stack starting with workspace: $(WORKSPACE_PATH)"
	@echo "Services will be available at:"
	@echo "  vLLM API:    http://localhost:$(PORT)/v1"
	@echo "  Open WebUI:  http://localhost:3000"
	@echo "  TT Monitor:  http://localhost:9090"
	@echo ""
	@echo "Run 'make logs-dev' to follow logs"

#==============================================================================
# Stop Targets
#==============================================================================

stop-prebuilt:
	@echo "Stopping prebuilt stack..."
	cd prebuilt && docker compose down

stop-source:
	@echo "Stopping source stack..."
	cd from_source && docker compose down

stop-dev:
	@echo "Stopping dev stack..."
	cd dev && docker compose down

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

test-dev: build-dev
	@echo "============================================"
	@echo "Testing dev variant"
	@echo "============================================"
	@if [ -z "$(WORKSPACE_PATH)" ]; then \
		echo "ERROR: WORKSPACE_PATH not set. Run 'make workspace-init' first."; \
		exit 1; \
	fi
	@# Ensure stopped first
	cd dev && docker compose down 2>/dev/null || true
	@# Copy env if needed
	@if [ ! -f dev/.env ]; then cp dev/.env.example dev/.env; fi
	@# Start stack
	cd dev && WORKSPACE_PATH=$(WORKSPACE_PATH) docker compose up -d
	@# Wait for healthy (longer timeout for dev builds)
	@echo "Waiting for vLLM to be healthy (timeout: $(WAIT_TIMEOUT)s)..."
	@./scripts/wait_for_healthy.sh $(WAIT_TIMEOUT) $(PORT)
	@# Run tests
	@echo ""
	@./tests/run_all.sh $(PORT) "$(MODEL)"
	@# Stop stack
	@echo ""
	@echo "Stopping stack..."
	cd dev && docker compose down
	@echo ""
	@echo "============================================"
	@echo "Dev tests complete"
	@echo "============================================"

#==============================================================================
# Device Targets (Galaxy, T3K, Blackhole, etc.)
#
# Override DEVICE_ENV, DEVICE_COMPOSE_EXTRA, DEVICE_PROJECT for your target.
# Example:
#   make run-device DEVICE_ENV=dev/.env.glm47.t3k DEVICE_COMPOSE_EXTRA="" DEVICE_PROJECT=glm-t3k
#==============================================================================

build-device:
	@docker ps >/dev/null 2>&1 || { echo "ERROR: 'docker ps' failed. Check Docker group membership."; exit 1; }
	@if [ ! -f "$(DEVICE_ENV)" ]; then \
		echo "ERROR: Device env file not found: $(DEVICE_ENV)"; exit 1; \
	fi
	@echo "Building dev image (env: $(DEVICE_ENV))..."
	docker compose --env-file $(DEVICE_ENV) -f dev/docker-compose.yml $(DEVICE_COMPOSE_EXTRA) -p $(DEVICE_PROJECT) build vllm-tt

run-device: build-device
	@echo "Starting container (project: $(DEVICE_PROJECT))..."
	docker compose --env-file $(DEVICE_ENV) -f dev/docker-compose.yml $(DEVICE_COMPOSE_EXTRA) -p $(DEVICE_PROJECT) up -d vllm-tt
	@echo ""
	@echo "Container: $(DEVICE_CONTAINER)"
	@echo "API: http://localhost:$(PORT)/v1"
	@echo ""
	@echo "Run 'make logs-device' to follow startup"
	@echo "Run 'make wait-device' to block until healthy"

stop-device:
	docker compose --env-file $(DEVICE_ENV) -f dev/docker-compose.yml $(DEVICE_COMPOSE_EXTRA) -p $(DEVICE_PROJECT) down

logs-device:
	docker compose --env-file $(DEVICE_ENV) -f dev/docker-compose.yml $(DEVICE_COMPOSE_EXTRA) -p $(DEVICE_PROJECT) logs -f vllm-tt

wait-device:
	@echo "Waiting for $(DEVICE_CONTAINER) to become healthy (timeout: $(DEVICE_WAIT_TIMEOUT)s)..."
	@elapsed=0; \
	while [ $$elapsed -lt $(DEVICE_WAIT_TIMEOUT) ]; do \
		status=$$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}no_healthcheck{{end}}' $(DEVICE_CONTAINER) 2>/dev/null || echo "not_found"); \
		case "$$status" in \
			healthy) echo "Healthy after $${elapsed}s"; exit 0 ;; \
			unhealthy) echo "UNHEALTHY — check 'make logs-device'"; exit 1 ;; \
			not_found) echo "Container $(DEVICE_CONTAINER) not found"; exit 1 ;; \
			no_healthcheck) echo "WARNING: No HEALTHCHECK defined, checking if running..."; \
				running=$$(docker inspect --format='{{.State.Running}}' $(DEVICE_CONTAINER) 2>/dev/null || echo "false"); \
				if [ "$$running" = "true" ]; then echo "Container running (no healthcheck)"; exit 0; fi; \
				echo "Container not running"; exit 1 ;; \
		esac; \
		if [ $$((elapsed % 300)) -eq 0 ] && [ $$elapsed -gt 0 ]; then \
			echo "  Still starting... ($${elapsed}s, status: $$status)"; \
		fi; \
		sleep 30; \
		elapsed=$$((elapsed + 30)); \
	done; \
	echo "Timed out after $(DEVICE_WAIT_TIMEOUT)s"; exit 1

verify-device:
	@echo "Verifying coherent output..."
	@response=$$(curl -sf --max-time 30 http://localhost:$(PORT)/v1/completions \
		-H 'Content-Type: application/json' \
		-d '{"model":"zai-org/GLM-4.7-Flash","prompt":"The capital of France is","max_tokens":10,"temperature":0}' 2>&1) || \
		{ echo "ERROR: API not responding (curl failed)"; exit 1; }; \
	echo "$$response" | python3 -c "$$( printf '%s\n' \
		'import sys, json' \
		'try:' \
		'    data = json.load(sys.stdin)' \
		'    text = data["choices"][0]["text"]' \
		'    print("Response:", text)' \
		'    assert "Paris" in text or len(text.split()) > 2, "Output may be garbled"' \
		'    print("Output looks coherent!")' \
		'except (json.JSONDecodeError, KeyError, IndexError) as e:' \
		'    print(f"ERROR: Unexpected API response format: {e}", file=sys.stderr)' \
		'    sys.exit(1)' \
		'except AssertionError as e:' \
		'    print(f"WARNING: {e}", file=sys.stderr)' \
		'    sys.exit(1)' \
	)" || { echo "WARNING: Verification failed — check 'make logs-device'"; exit 1; }

diagnostics-device:
	@echo "=== TT Device Diagnostics ==="
	@echo ""
	@echo "Step 1/4: Stopping ALL vllm-tt containers..."
	@# Stop containers from both default (dev) and named project
	-docker compose --env-file $(DEVICE_ENV) -f dev/docker-compose.yml $(DEVICE_COMPOSE_EXTRA) down 2>/dev/null
	-docker compose --env-file $(DEVICE_ENV) -f dev/docker-compose.yml $(DEVICE_COMPOSE_EXTRA) -p $(DEVICE_PROJECT) down 2>/dev/null
	-docker ps -q --filter "name=vllm-tt" | xargs -r docker stop 2>/dev/null
	@echo "Clearing stale UMD device locks..."
	-rm -f /dev/shm/TT_UMD_LOCK.* 2>/dev/null || true
	@echo ""
	@echo "Step 2/4: Resetting devices (clean state)..."
	-tt-smi -glx_reset 2>/dev/null || tt-smi -r 2>/dev/null || echo "WARNING: Device reset not available"
	-rm -f /dev/shm/TT_UMD_LOCK.* 2>/dev/null || true
	@echo ""
	@echo "Step 3/4: Running diagnostics (Metal + Inspector + triage)..."
	docker compose --env-file $(DEVICE_ENV) -f dev/docker-compose.yml $(DEVICE_COMPOSE_EXTRA) -f dev/docker-compose.diagnostics.yml \
		-p $(DEVICE_PROJECT) run --rm vllm-tt
	@echo ""
	@echo "Step 4/4: Resetting devices (clean state for next run)..."
	-rm -f /dev/shm/TT_UMD_LOCK.* 2>/dev/null || true
	-tt-smi -glx_reset 2>/dev/null || tt-smi -r 2>/dev/null || echo "WARNING: Device reset not available"
	-rm -f /dev/shm/TT_UMD_LOCK.* 2>/dev/null || true
	@echo ""
	@echo "=== Diagnostics complete. Devices reset. ==="
	@echo "Run 'make run-device' to start vLLM."

shell-device:
	@echo "=== TT Tools Interactive Shell ==="
	@echo ""
	docker compose --env-file $(DEVICE_ENV) -f dev/docker-compose.yml $(DEVICE_COMPOSE_EXTRA) -f dev/docker-compose.shell.yml \
		-p $(DEVICE_PROJECT) run --rm vllm-tt

# Shortcuts — Galaxy Wormhole (default config)
run-galaxy: ; $(MAKE) run-device
stop-galaxy: ; $(MAKE) stop-device
logs-galaxy: ; $(MAKE) logs-device
wait-galaxy: ; $(MAKE) wait-device
verify-galaxy: ; $(MAKE) verify-device
diagnostics-galaxy: ; $(MAKE) diagnostics-device
shell-galaxy: ; $(MAKE) shell-device

#==============================================================================
# Utility Targets
#==============================================================================

logs-prebuilt:
	cd prebuilt && docker compose logs -f

logs-source:
	cd from_source && docker compose logs -f

logs-dev:
	cd dev && docker compose logs -f

#==============================================================================
# Workspace Targets
#==============================================================================

workspace-init:
	@echo "Initializing workspace infrastructure..."
	@if [ ! -f scripts/workspace.env ]; then \
		echo "Creating scripts/workspace.env from example..."; \
		cp scripts/workspace.env.example scripts/workspace.env; \
		echo ""; \
		echo "IMPORTANT: Edit scripts/workspace.env to set your fork URLs before running again."; \
		exit 1; \
	fi
	./scripts/workspace.sh init

workspace-create:
	@if [ -z "$(NAME)" ]; then \
		echo "Usage: make workspace-create NAME=<workspace-name>"; \
		exit 1; \
	fi
	./scripts/workspace.sh create $(NAME)

workspace-list:
	./scripts/workspace.sh list

workspace-delete:
	@if [ -z "$(NAME)" ]; then \
		echo "Usage: make workspace-delete NAME=<workspace-name>"; \
		exit 1; \
	fi
	./scripts/workspace.sh delete $(NAME)

workspace-status:
	./scripts/workspace.sh status $(NAME)

workspace-sync:
	./scripts/workspace.sh sync

reset-devices:
	@echo "Resetting TT devices..."
	@device_ids=$$(tt-smi -ls 2>/dev/null | grep -oP 'Device \K[0-9]+' | paste -sd, -); \
	if [ -n "$$device_ids" ]; then \
		echo "  Devices: $$device_ids"; \
		tt-smi -r $$device_ids || echo "Warning: Device reset may have partially failed"; \
	else \
		echo "  No devices found via tt-smi, trying reset all..."; \
		tt-smi -r || echo "Warning: Device reset may have partially failed"; \
	fi
	@echo "Waiting for devices to reinitialize..."
	@sleep 5
	@echo "Device reset complete"

clean:
	@echo "Stopping all containers..."
	cd prebuilt && docker compose down --rmi local 2>/dev/null || true
	cd from_source && docker compose down --rmi local 2>/dev/null || true
	cd dev && docker compose down --rmi local 2>/dev/null || true
	@echo "Removing images..."
	docker rmi vllm-tt:latest vllm-tt-source:latest vllm-tt-dev:latest tt-monitor:latest 2>/dev/null || true
	@echo "Clean complete"
