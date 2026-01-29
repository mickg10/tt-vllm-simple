# TT vLLM Simple

Docker setup for running vLLM with Tenstorrent hardware acceleration.

## Project Structure
- `Dockerfile` - Pre-built image (fast, uses ghcr.io/tenstorrent/tt-inference-server)
- `fullbuild/Dockerfile` - Full source build (slow, builds tt-metal + vLLM from source)
- `docker-compose.yml` - Service orchestration (vLLM, Open WebUI, TT-Monitor)
- `tt-monitor/` - Device monitoring dashboard
- `entrypoint.sh` - Container startup script

## Services & Ports
- vLLM API: http://localhost:8088 (OpenAI-compatible)
- Open WebUI: http://localhost:3000 (Chat interface)
- TT-Monitor: http://localhost:9090 (Device dashboard)

## Quick Start
docker compose up -d

## Environment Variables
- HF_MODEL: Model to serve (default: Qwen/Qwen3-0.6B)
- HF_TOKEN: Hugging Face access token
- VLLM_RPC_TIMEOUT: RPC timeout in ms
- MESH_DEVICE: Device config (N150, N300, T3K, TG)

## Upstream Repositories
- tt-metal: https://github.com/tenstorrent/tt-metal (branch: main)
- vLLM fork: https://github.com/tenstorrent/vllm (branch: dev)

## Build Commands
# Pre-built (fast)
docker compose build

# Full source build
docker build -t vllm-tt-fullbuild:latest ./fullbuild

## Common Tasks
- Start services: docker compose up -d
- View logs: docker compose logs -f vllm-tt
- Run benchmark: python benchmark_forever.py
- Monitor devices: http://localhost:9090
