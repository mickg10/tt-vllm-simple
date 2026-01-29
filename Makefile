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
