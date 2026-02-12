# GLM-4.7-Flash on Tenstorrent: Plan, Architecture & History

Date started: 2026-02-03
Last updated: 2026-02-11

Target model: `zai-org/GLM-4.7-Flash` (HuggingFace `model_type=glm4_moe_lite`)
Hardware: Tenstorrent Wormhole (T3K — 1x8 mesh, 8 devices)
Primary constraint: **Do not break** `Qwen/Qwen3-32B` serving/inference.

---

## Table of Contents

1. [Model Architecture](#1-model-architecture)
2. [Development Infrastructure](#2-development-infrastructure)
3. [Phase Plan](#3-phase-plan)
4. [Execution History](#4-execution-history)
5. [Bugs Found and Fixed](#5-bugs-found-and-fixed)
6. [Performance Optimization](#6-performance-optimization)
7. [Current Status](#7-current-status)
8. [Next Steps](#8-next-steps)
9. [Key Technical Decisions](#9-key-technical-decisions)
10. [Risk Register](#10-risk-register)
11. [Reference Material](#11-reference-material)

---

## 1. Model Architecture

### 1.1 GLM-4.7-Flash Overview

GLM-4.7-Flash is a Mixture-of-Experts (MoE) model with DeepSeek-style Multi-Latent
Attention (MLA). It is the first MoE + MLA model ported to TT hardware.

Core parameters:
- `vocab_size=154880`, `hidden_size=2048`
- `num_hidden_layers=47` (layers 0-46; layer 47 in checkpoint is MTP-related, ignored)
- `intermediate_size=10240` (dense MLP in layer 0)
- `num_attention_heads=20`

### 1.2 Multi-Latent Attention (MLA)

MLA compresses KV projections through a low-rank bottleneck:

```
Query path:   x -> q_a_proj(768) -> RMSNorm -> q_b_proj(20*256) -> split(nope|rope)
KV path:      x -> kv_a_proj(576) -> split(kv_lora|rope)
Cache:        stores compressed KVPE vector (576-dim) per token, not full K/V
Attention:    FlashMLA operates in KVPE space (576-dim last axis)
Output:       kv_b2 decompresses from kv_lora_rank to v_head_dim
```

Key dimensions:
| Parameter | Value | Notes |
|-----------|-------|-------|
| `q_lora_rank` | 768 | Query LoRA bottleneck |
| `kv_lora_rank` | 512 | KV LoRA bottleneck |
| `qk_nope_head_dim` | 192 | Non-RoPE Q/K slice |
| `qk_rope_head_dim` | 64 | RoPE-applied Q/K slice |
| `qk_head_dim` | 256 | = nope + rope |
| `v_head_dim` | 256 | Value dimension per head |
| `kvpe_dim` | 576 | = kv_lora_rank(512) + rope(64); what gets cached |

KV cache shape (paged): `[num_blocks, 1, block_size, 576]`
- Single "head" because MLA compresses all heads into one latent vector per token
- `head_size=576` (KVPE), `num_kv_heads=1`

### 1.3 Mixture of Experts (MoE)

- Layer 0: dense MLP (`first_k_dense_replace=1`)
- Layers 1-46: MoE with 64 routed experts + 1 shared expert
- Top-k routing: 4 experts per token
- Router: sigmoid scoring + `e_score_correction_bias` + top-k
- `routed_scaling_factor=1.8`
- `moe_intermediate_size=1536` (per expert)
- `topk_method="noaux_tc"` (no auxiliary loss, top-compact selection)
- `norm_topk_prob=True`

### 1.4 Checkpoint Nuance

- Checkpoint includes `model.layers.47.*` (MTP / next-n predict related)
- Transformers reference ignores these for baseline generation
- Our implementation ignores layer 47 — production scope is baseline generation only

---

## 2. Development Infrastructure

### 2.1 Repository Structure

Three repos coordinated via bare git repos + worktrees:

| Repo | Branch | Location | Purpose |
|------|--------|----------|---------|
| `docker_tt` | `glm47_flash` | `ws/glm47_flash/docker_tt/` | Docker orchestration, env files, tests |
| `tt-metal` | `glm47_flash` | `ws/glm47_flash/tt-metal/` | TT model code (the actual implementation) |
| `vllm` | `glm47_flash` | `ws/glm47_flash/vllm/` | vLLM config shim, platform registration |

Bare repos at `/home/ttuser/src_docker/{docker_tt,tt-metal,vllm}.git`

GitHub forks:
- docker_tt: https://github.com/mickg10/tt-vllm-simple/tree/glm47_flash
- tt-metal: https://github.com/mickg10/tt-metal/tree/glm47_flash
- vllm: https://github.com/mickg10/vllm/tree/glm47_flash

### 2.2 Running the Dev Stack

```bash
cd /home/ttuser/src_docker/ws/glm47_flash/docker_tt
docker compose --env-file dev/.env.glm47 -f dev/docker-compose.yml \
  up -d --force-recreate vllm-tt
```

Two env configurations:
- `dev/.env.glm47` — perf-trace-tp config (~6 tok/s, coherent, tracing + TP enabled)
- `dev/.env.glm47.correctness` — correctness-first (~3 tok/s, hifi4 math, no tracing)

### 2.3 Reference Endpoint

- GPU reference: `http://localhost:8087/v1` (NVIDIA GPU running full `zai-org/GLM-4.7-Flash`)
- TT endpoint: `http://localhost:8088/v1` (TT hardware)
- Comparison via benchmark harness: `scripts/run_perf_iteration.py`

### 2.4 Key File Locations (tt-metal)

```
models/demos/glm4_moe_lite/tt/
├── generator_vllm.py      # vLLM model interface (545 lines)
├── model_tt.py             # Core TT runner (1503 lines)
├── decoder_layer_tt.py     # Per-layer decode/prefill (1338 lines)
├── layer_weights.py        # Weight conversion + TP sharding (639 lines)
├── moe_tt.py               # MoE router + sparse/dense experts (1102 lines)
├── config.py               # Hyperparameter dataclass (128 lines)
├── tt_embedding.py          # TT embedding lookup (75 lines)
├── weights.py              # Lazy safetensors loading (115 lines)
├── debug_runtime.py        # Dealloc disable for debugging (37 lines)
├── layer0_tt.py            # Layer-0 bring-up harnesses (1690 lines)
├── reference_layer0.py     # CPU reference for layer 0 (187 lines)
└── reference_moe.py        # CPU reference for MoE (201 lines)
```

### 2.5 Key File Locations (vLLM)

- Config shim: `vllm/transformers_utils/configs/glm4_moe_lite.py` (125 lines)
- Platform registration: `vllm/platforms/tt.py` (registers `TTGlm4MoeLiteForCausalLM`)

---

## 3. Phase Plan

### Phase 1: Baselines and Guardrails

**Goal:** Establish Qwen32B backtest contract, capture runtime state, stabilize workspace.

Gates:
- Gate A: Qwen/Qwen3-32B E2E works (health, model list, chat completion)
- Gate B: No silent dependency drift (any pip change needs justification + backtest)

Tasks:
- [x] Snapshot baseline environment (tt-metal SHA, vllm SHA, transformers version)
- [x] Create workspace `glm47_flash` with matching branches in all 3 repos
- [x] Add Qwen32B smoke test script
- [x] Verify `dev` Docker variant boots and serves Qwen

### Phase 2: vLLM Enablement for `glm4_moe_lite`

**Goal:** Make vLLM recognize and configure GLM-4.7-Flash correctly.

Tasks:
- [x] Add `Glm4MoeLiteConfig` via `_CONFIG_REGISTRY` (stay on Transformers 4.57.1)
- [x] Set `use_mla=True`, `head_size=576` (KVPE), `num_kv_heads=1`
- [x] Register `TTGlm4MoeLiteForCausalLM` architecture in `platforms/tt.py`
- [x] Add `glm4_moe_lite` to MLA model list in vLLM config
- [x] Verify: vLLM server boots with GLM config, allocates correct KV cache shape

### Phase 3: tt-metal Skeleton + vLLM Integration

**Goal:** Model code exists, server boots for GLM without crashes.

Tasks:
- [x] Create `models/demos/glm4_moe_lite/tt/` directory
- [x] Implement `generator_vllm.py` with vLLM interface contract
- [x] Implement weight loading via `LazyStateDict` (reuse DeepSeek pattern)
- [x] Implement `allocate_kv_cache()` with MLA-aware shape
- [x] Placeholder forward returns zeros — server boots

### Phase 4: Layer 0 Correctness (Prefill)

**Goal:** Single-layer prefill produces correct hidden states.

Tasks:
- [x] Implement embedding lookup (`tt_embedding.py`)
- [x] Implement RMSNorm (reuse `ttnn.rms_norm`)
- [x] Implement MLA attention projections (q_a, q_b, kv_a, kv_b1, kv_b2, output)
- [x] Implement RoPE (partial rotation on rope slice only)
- [x] Implement FlashMLA prefill attention
- [x] Implement dense MLP (layer 0 only)
- [x] Build CPU reference for layer 0 (`reference_layer0.py`)
- [x] Validate: layer 0 prefill hidden states match CPU reference within BF16 tolerance

### Phase 5: KV Cache + Decode Correctness

**Goal:** Paged KV cache fill/update works, decode produces correct tokens.

Tasks:
- [x] Implement `paged_fill_cache` for prefill (KVPE layout)
- [x] Implement `paged_update_cache` for decode
- [x] Implement FlashMLA decode attention (`paged_flash_multi_latent_attention_decode`)
- [x] Implement decode Q path with LoRA decomposition
- [x] Build debug harnesses for unpaged vs paged decode comparison
- [x] Validate: token-by-token decode matches expected behavior for short prompts

### Phase 6: Full Model (All 47 Layers + MoE)

**Goal:** End-to-end inference with all layers, including MoE routing and expert execution.

Tasks:
- [x] Extend decoder layer across all 47 layers
- [x] Implement MoE router (`moe_topk_tt` — sigmoid + bias + topk on device)
- [x] Implement sparse expert execution (`ttnn.sparse_matmul`)
- [x] Implement shared expert path
- [x] Implement expert output merge (routed + shared + residual)
- [x] Implement LM head (with optional vocab-sharded TP)
- [x] Implement on-device sampling (greedy)
- [x] Build CPU reference MoE router for validation (`reference_moe.py`)
- [x] Validate: end-to-end chat produces coherent English
- [x] Validate: deterministic at temperature=0
- [x] Validate: no KV-boundary corruption

### Phase 7: Performance + Productionization (In Progress)

**Goal:** Close throughput gap from ~6 tok/s to target 30 tok/s.

Tasks:
- [x] Add per-stage decode profiling infrastructure
- [x] Add benchmark harness with warmup support
- [x] Fix decode lane inflation in sparse MoE
- [x] Add sparse MoE BF16-speed kernel defaults
- [x] Implement TP sharding for dense path (partial — row-parallel attention + MLP)
- [x] Implement fused Q+KV-A projection (`w_q_kv_a`)
- [x] Enable decode tracing (`trace_mode=decode_only`)
- [x] Add thinking-disable chat template (UX improvement)
- [x] Add tool-call parser support (`glm47`)
- [ ] Full TP sharding for MoE experts (not just dense path)
- [ ] Move decode hot tensors to L1/sharded memory
- [ ] Enable prefix caching for GLM
- [ ] Native FlashMLA prefill (currently using decode-loop fallback)
- [ ] 8-bit checkpoint ingestion
- [ ] OpenCode coding suite gate (blocked on task completion quality)

---

## 4. Execution History

### 2026-02-03: Project Kickoff

- Created overall plan with 7 phases, risk register, and fallback ladder
- Identified key risks: MLA correctness, vLLM feature gaps, paged KV cache stability
- Established hard constraints: Qwen32B non-regression, no CPU fallback for production

### 2026-02-04: Phase 1 — Baselines

- Snapshotted baseline: tt-metal `983e2105`, vllm `3499ffa`, transformers `4.57.1`
- Created `glm47_flash` workspace and branches in all 3 repos
- Established Qwen32B backtest contract
- Verified GLM model already downloaded to HF cache (~19GB)

### 2026-02-04 to 2026-02-05: Phases 2-3 — vLLM Config + tt-metal Skeleton

- Added `Glm4MoeLiteConfig` to vLLM's config registry (avoids Transformers 5 dependency)
- Registered TT model architecture in platform detection
- Created tt-metal model scaffold with weight loading, KV cache allocation
- Server boots for GLM without crashes (placeholder forward)

Key commit: `73864ced3c` (tt-metal: Add GLM4 MoE Lite demo scaffold)

### 2026-02-05 to 2026-02-06: Phase 4 — Layer 0 Correctness

- Implemented full MLA attention pipeline: q_a → RMSNorm → q_b → split(nope|rope) → etc.
- Built CPU reference for layer 0 validation
- Discovered RoPE partial rotation subtlety (only 64-dim rope slice, rest is nope)
- Validated layer 0 prefill against CPU reference within BF16 tolerance

### 2026-02-06: Phase 5 — KV Cache + Decode

- Implemented paged KVPE cache fill and update
- Implemented FlashMLA decode attention
- Built debug harnesses comparing unpaged vs paged decode
- Short-prompt decode producing tokens (but with issues at KV block boundaries — later fixed)

### 2026-02-06: Phase 6 — Full Model

- Extended all 47 layers, implemented MoE router + sparse experts
- First end-to-end chat producing coherent English
- Initial throughput: ~2.1 tok/s (vs 30 tok/s target)

Key commits:
- `150995d339` (tt-metal: vllm runner bring-up fixes and profiling)
- `3d72cdc5ff` (tt-metal: fix KVPE cache shape in vLLM adapter)
- `fe3586363` (vllm: GLM defaults, parsers, and config plumbing)
- `33eb401` (docker_tt: add docker bring-up envs, benchmarks, smoke tests)

### 2026-02-06: Manual Validation Gate — PASS

- `/health`, `/v1/models` pass
- Open WebUI on `:3000` works
- Tool-calling (stream + non-stream) returns valid `tool_calls`
- Quality check: short strict prompt returns `Blue` (no reasoning leakage)
- Thinking-disable chat template halved latency (9.3s → 2.8s for short prompts)

### 2026-02-06: Phase 7 Begins — Performance Work

First profiling run established the decode stage breakdown:

| Stage | ms/tok | % of total |
|-------|--------|------------|
| `moe_experts` | 113-116 | 26% |
| `q_path` | 75-77 | 18% |
| `kv_cache_update` | 56-59 | 13% |
| `moe_router` | 56-58 | 13% |
| `moe_shared` | 43-44 | 10% |
| `attn_out` | 42-43 | 10% |
| **Total** | **430-441** | **~2.3 tok/s** |

### 2026-02-06: Decode Lane Inflation Discovery

Sparse MoE debug probe revealed a structural inefficiency:
- Single-request decode was padded to `tokens_per_device=32` across 8 devices = 256 total tokens
- Actual active tokens: 1
- This inflated MoE compute by ~32x for decode

**Fix:** Changed sparse padding from hardcoded 32 to `ceil(block_size / gcd(block_size, dispatch_devices))` — minimum legal sparse block requirement.

After fix: `tokens_per_device=4`, `total_tokens=32` (still padded but 8x less).

### 2026-02-06: Negative Experiments (Rejected)

| Experiment | Result | Decision |
|------------|--------|----------|
| Dense BF8 weights | 1.91 tok/s (slower than BF16 2.64) | Rejected, keep BF16 |
| `dense_decode` experts impl | Unstable startup, container restarts | Rejected |
| MLA `hifi2` fidelity | Timeout on same test case | Rejected, keep `hifi4` |
| CPU router fallback | 1.86 tok/s (slower than TT 2.60) | Rejected |

### 2026-02-07 to 2026-02-08: TP Sharding + Trace Mode

- Implemented tensor-parallel sharding for dense path:
  - Row-parallel attention output projection
  - Column/row-parallel MLP (gate_up / down)
  - Vocab-sharded LM head with host-side reduction
- Enabled decode tracing (`trace_mode=decode_only`, `trace_region_size=40MB`)
- Decode throughput jumped from ~2.6 tok/s to ~6 tok/s (decode TPS)

Key commits:
- `6209f449c5` (tt-metal: mitigate TTNN view dealloc corruption)
- `b1fdf7c85f` (tt-metal: fix GLM decode trace RoPE host->device corruption)
- `ee73915cb8` (tt-metal: block decode trace replay for correctness)

### 2026-02-08 to 2026-02-09: Fused QKV-A + Router on Device

- Fused `w_q_a` + `w_kv_a` into single matmul (`GLM4_MOE_LITE_FUSE_QKV_A=1`)
- Moved MoE router from CPU to TT device (`GLM4_MOE_LITE_MOE_ROUTER_IMPL=tt`)
- Throughput (decode TPS, warmed): ~14 tok/s peak (best iteration `i001_router_tt`)
- But e2e TPS lower due to slow prefill (still using decode-loop path, ~20s TTFT)

### 2026-02-09: Trace Correctness Fix

- Discovered: non-blocking trace replay in `decode_loop_trace` prefill path caused overlapping
  trace replays to corrupt persistent trace inputs / KV updates → gibberish output
- Fix: made trace replay blocking in `_decode_trace_sampling`

Key commit: `ee73915cb8` (tt-metal: block decode trace replay for correctness)

### 2026-02-10: Correctness Regression — Nondeterministic Decode

- At `temperature=0`, same prompt could produce different outputs across runs
- Root cause: FlashMLA decode with `fp32_dest_acc_en=True` corrupts greedy decode
  at the first KV block boundary (`pos == block_size = 64`)
- This was a latent bug exposed by longer sequences

### 2026-02-10 to 2026-02-11: FlashMLA fp32 Fix

- Root cause confirmed: `fp32_dest_acc_en=True` in FlashMLA decode corrupts at KV page boundary
- Fix: force `fp32_dest_acc_en=False` unless explicitly overridden via `GLM4_MOE_LITE_UNSAFE_ALLOW_FP32_MLA`
- Added boundary regression test
- Determinism probe: 5 sequential repeats, identical output — **PASS**

Key commits:
- `3b63e3cc34` (tt-metal: avoid FlashMLA KV-boundary corruption)
- `b2fbf06a6` (vllm: add optional GLM page_table boundary logs)

### 2026-02-11 to 2026-02-12: Sprint 3 — Flash Prefill, MoE Fix, LoFi

Sprint 3 focused on three areas: native prefill, MoE stability, and compute config.

**Native FlashMLA prefill (`flash_mla_prefill`):**
- Implemented `flash_mla_prefill` kernel for prefill path (replaces iterative decode loop)
- Added trace release/re-capture lifecycle for prefill ↔ traced decode coexistence
- TTFT improved dramatically: 2.4s (short), 5.4s (68 tok) vs previous 20-45s

**MoE sparse_matmul chunking fix:**
- Root cause: `sparse_matmul` with `per_core_M=1` only supports 1 sparsity block (32 tokens)
- Prompts > 32 tokens would hang in the MoE prefill path
- Fix: automatic chunking when `total_tokens > sparsity_block_size`
- Validated: 50-token and 97-token prompts now work correctly

**LoFi + packer_l1_acc for decode:**
- Applied DeepSeek V3 compute kernel pattern to all MLP/MoE linear ops
- `MathFidelity.LoFi`, `math_approx_mode=True`, `packer_l1_acc=True`, `fp32_dest_acc_en=False`
- Result: 218ms → 195ms per token (~11% improvement)
- Correctness verified (7*8=56, capital of France=Paris)

**RFC-1 (L1 MoE + fused gate_up) — from Sprint 1:**
- `GLM4_MOE_LITE_EP_L1=1`: L1 memory for MoE decode experts (+4%)
- `GLM4_MOE_LITE_FUSE_EXPERTS_GATE_UP=1`: Fused w1+w3 expert projections

Sprint 3 results (1k context, 500 gen):
| Batch | Aggregate tok/s | Per-user tok/s | TTFT | ITL |
|-------|----------------|----------------|------|-----|
| 1 | 4.5 | 4.5 | 59s | 223ms |
| 4 | 14.6 | 4.5 | 25s | 223ms |
| 8 | 28.5 | 4.5 | 30s | 221ms |
| 32 | TBD | ~4.5 | TBD | ~221ms |

**Key observation:** Per-user decode speed is constant at ~4.5 tok/s regardless of batch size.
This means aggregate throughput scales linearly: bs=32 should yield ~144 tok/s aggregate.

### 2026-02-12: Ralph Loop Start — Performance Optimization Sprint

Targets updated per user requirements:
- **Batch=1 decode:** 30 tok/s (currently 4.5 tok/s, need 6.7x)
- **Batch=32 decode:** 140+ tok/s aggregate (currently ~144 predicted, may already meet!)
- **Benchmark matrix:** (1k/500, 10k/1000, 29k/3000 ctx/gen) × (batch=1,4,8,32)

Team structure:
- Team lead / architect (coordinates, consults Codex gpt-5.2)
- Implementer (makes code changes, feature-flags everything)
- Tester (coherency verification at tiny sizes FIRST)
- Benchmarker (full matrix, records to perf-opt.md)

**Experiment 1: L1 WIDTH_SHARDED decode activations (REJECTED)**
- Used `ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG` sentinel for MLP activations
- Result: zero improvement (222.9ms ITL before and after)
- Root cause: sentinel constant doesn't specify shard specs or change matmul program config;
  all three components (DRAM-sharded weights, program config, activation shards) must be co-designed
- Reverted

**Experiment 2: DRAM-sharded weights Phase 1 (attention linears)**
- Implemented `dram_sharded_weight_config()` + `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`
  for 5 attention projections: w_q_a, w_q_b, w_kv_a, w_q_kv_a (fused), w_o
- Feature flag: `GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1`
- Coherency: 30/32 PASS (no regression)
- **bs=32 aggregate: 38.0 tok/s (+37% from 27.8 baseline)**
- bs=32 decode loop: 311s vs 468s baseline (-34%)
- bs=1: 3.5 tok/s decode (227ms ITL) — slight regression from baseline 4.5 (overhead of resharding)
- bs=4: 14.1 tok/s aggregate (comparable to 14.6 baseline)
- Phase 1 scope (attention only ~34% of decode) mainly helps batched throughput
- Phase 2 planned: MLP + shared expert + per-head weights for per-user latency gains

Process: implement → restart → verify coherency → benchmark → commit + push

---

## 5. Bugs Found and Fixed

### 5.1 FlashMLA KV-Boundary Corruption (Critical)

**Symptom:** Greedy decode corrupts exactly when the 2nd paged-KV block is first touched
(`pos == block_size`, typically at position 64).

**Root cause:** FlashMLA decode with `fp32_dest_acc_en=True` has a compute kernel bug
that manifests at page boundaries. The fp32 accumulation interacts incorrectly with the
KV cache page crossing logic.

**Fix:** Force `fp32_dest_acc_en=False` in FlashMLA for GLM unless explicitly overridden
via unsafe escape hatch. This trades theoretical precision for correct behavior.

**Impact:** Was the primary correctness blocker. All quality validation depends on this fix.

### 5.2 MoE Decode Corruption (View/UAF)

**Symptom:** MoE sparse path produced wrong expert outputs during decode.

**Root cause:** Permute output in sparse MoE path was a view, and subsequent operations
could alias/overwrite the underlying buffer (use-after-free semantics in TTNN).

**Fix:** Clone permute output before further processing.

**Commit:** `500f2e15f9`

### 5.3 TTNN View Dealloc Corruption

**Symptom:** Intermittent garbage in decode path.

**Root cause:** TTNN's `deallocate()` on a view could invalidate the parent tensor's memory.

**Fix:** Added mitigation in the GLM code path to avoid deallocating views.
Debug escape hatch: `GLM4_MOE_LITE_DISABLE_DEALLOC=1` disables all explicit deallocation.

**Commit:** `6209f449c5`

### 5.4 Decode Trace RoPE Host→Device Corruption

**Symptom:** RoPE values corrupted when running decode under tracing.

**Root cause:** RoPE tensors were being prepared on host and sent to device in a way that
conflicted with trace capture/replay semantics.

**Fix:** Ensure RoPE inputs are stable device tensors before trace capture.

**Commit:** `b1fdf7c85f`

### 5.5 Non-Blocking Trace Replay Race

**Symptom:** Gibberish output during `decode_loop_trace` prefill (iterative decode path).

**Root cause:** Non-blocking trace replay allowed overlapping replays to race on persistent
trace inputs and KV cache updates.

**Fix:** Made trace replay blocking in `_decode_trace_sampling`.

**Commit:** `ee73915cb8`

### 5.6 Dense-Decode Mesh Expert Slicing

**Symptom:** Rank-5 tensor error in MoE dense expert path on MeshDevice.

**Root cause:** Expert weight slicing assumed single-device tensor rank.

**Fix:** Handle mesh-aware slicing in `moe_dense_experts_forward_decode_tt`.

**Commit:** `5ad783dec8`

---

## 6. Performance Optimization

### 6.1 Throughput Timeline

| Date | Config | Decode TPS | E2E TPS | Key Change |
|------|--------|-----------|---------|------------|
| Feb 6 | Baseline (no trace, no TP) | 2.44 | 2.44 | First working model |
| Feb 6 | + Lane inflation fix | 2.55 | 2.53 | Sparse MoE padding reduced |
| Feb 6 | + Sparse BF16 kernels | 2.60 | 2.60 | hifi2 + approx=1 |
| Feb 8 | + Decode tracing | 6.48 | 5.47 | `trace_mode=decode_only` |
| Feb 8 | + TP sharding (dense) | 6.04 | 4.73 | Row-parallel attention/MLP |
| Feb 9 | + Fused QKV-A | ~6.5 | ~5.4 | Single matmul for q+kv projection |
| Feb 9 | + TT router | 14.4 | 13.6 | On-device routing (peak) |
| Feb 10 | + Blocking trace fix | 6.4 | 5.3 | Correctness fix reduced peak |
| Feb 10 | + packer_l1_acc | 13.4 | 13.4 | L1 accumulation enabled |
| **Feb 11** | **Current best (perf-trace-tp)** | **~6.4** | **~5.3** | **Stable, coherent config** |

Note: Some iterations showed ~14 tok/s decode but with unreliable TTFT measurement
(streaming content issue). The **stable, measured, coherent** config is ~6 tok/s decode.

### 6.2 Decode Stage Profile (Current)

From profiling with `GLM4_MOE_LITE_PROFILE=1`, at ~2.4 tok/s baseline (pre-trace):

| Stage | ms/tok | Optimization Applied |
|-------|--------|---------------------|
| `moe_experts` | 110 | Lane fix, sparse BF16 kernels |
| `q_path` | 71 | Fused QKV-A (partial) |
| `moe_router` | 58 | Moved to TT device |
| `kv_cache_update` | 49 | — |
| `moe_shared` | 45 | — |
| `attn_out` | 43 | TP row-parallel |
| `moe_merge` | 9 | — |
| **Total** | **~411** | |

With tracing, the aggregate drops to ~155 ms/tok (~6.4 tok/s decode).

### 6.3 Root Cause of Remaining Gap

The gap from ~6 tok/s to target 30 tok/s is structural:

1. **MoE experts not TP-sharded** — experts run replicated on each device; no expert parallelism
2. **Prefill uses decode-loop** — iterative per-token prefill via traced decode path, not native
   FlashMLA prefill; TTFT is ~20s for medium prompts
3. **L1/sharded memory not used** — decode hot tensors are in DRAM, not L1
4. **No prefix caching** — repeated-prefix workloads pay full prefill every time
5. **Single-sequence batch** — `MAX_NUM_SEQS=1` for stability during bring-up

### 6.4 Negative Experiments (Rejected from Baseline)

| Experiment | Throughput | Issue | Verdict |
|------------|-----------|-------|---------|
| Dense BF8 weights | 1.91 tok/s | Slower than BF16 (2.64) | Keep BF16 |
| `dense_decode` experts | N/A | Unstable startup, restarts | Not viable |
| MLA `hifi2` fidelity | N/A | Timeout on test case | Keep `hifi4` |
| CPU router | 1.86 tok/s | Slower than TT router (2.60) | Keep TT router |

---

## 7. Current Status

### 7.1 Correctness

| Check | Status | Evidence |
|-------|--------|----------|
| Manual chat (coherent English) | **PASS** | Interactive testing via WebUI + API |
| Determinism (temperature=0, 5 repeats) | **PASS** | Artifact: `determinism_probe_zai-org_GLM-4.7-Flash_20260211_133728.md` |
| KV-boundary (pos >= 64) | **PASS** | FlashMLA fp32 fix + regression test |
| Qwen32B non-regression | **PASS** | `qwen32b_smoke.sh` after every change |

### 7.2 Performance

| Endpoint | Decode TPS | E2E TPS | TTFT |
|----------|-----------|---------|------|
| GPU reference (:8087) | ~47 | ~42 | ~0.6s |
| TT perf-trace-tp (:8088) | ~4.5/user | ~28.5 agg (bs=8) | ~2.4s (short) |
| TT correctness (:8088) | ~2.7 | ~2.4 | ~39s |
| Qwen32B TT (:8088) | ~18.9 | ~18.8 | ~0.2s |

**Latest benchmark (1k ctx, 500 gen, 2026-02-12):**
- Per-user decode: 4.5 tok/s (223ms ITL) — constant across batch sizes 1-8
- Aggregate scales linearly: 4.5 (bs=1), 14.6 (bs=4), 28.5 (bs=8)
- TTFT: 2.4s (short prompts) to 59s (1k tokens, includes trace re-capture)

**Post DRAM-sharded Phase 1 (attention linears only, 2026-02-12):**
- bs=1: 3.5 tok/s decode (227ms ITL, 8.6s TTFB) — slight regression, overhead of activation resharding
- bs=4: 14.1 agg tok/s (4.1 per-user, 226.8ms ITL) — comparable to baseline
- **bs=32: 38.0 agg tok/s** (4.0 per-user, 198ms ITL, 421s wall) — **+37% from 27.8 baseline**
- Decode loop at bs=32: 311s vs 468s baseline (-34% reduction)
- Coherency: 30/32 PASS (no regression)
- Phase 2 (MLP + MoE) expected to improve per-user latency further

### 7.3 Environment Configurations

**Perf-trace-tp** (`dev/.env.glm47`) — the primary config:
- `trace_mode=decode_only`, `trace_region_size=40MB`
- `enable_model_warmup=true`, `sample_on_device_mode=decode_only`
- TP=1, fused QKV-A=1, V-cache-slice=1
- Sparse experts: hifi2, no fp32 acc, approx=1
- MLA: hifi2, approx=1
- Dense weights: BF16, Expert weights: BF8, KV cache: BF8

**Correctness** (`dev/.env.glm47.correctness`) — for debugging:
- `trace_mode=none`, no warmup
- TP=0, no fused QKV-A
- Sparse experts: hifi4, fp32 acc, no approx
- MLA: hifi4, fp32 acc
- CPU router fallback available

### 7.4 Git SHAs (Latest)

| Repo | SHA | Description |
|------|-----|-------------|
| `docker_tt` | `0255626` | WORKLOG update, env consolidation |
| `tt-metal` | `3b63e3cc34` | FlashMLA fp32 dest acc safety gate |
| `vllm` | `b2fbf06a6` | Optional page_table boundary logging |

---

## 8. Next Steps

### 8.1 Performance Roadmap (Prioritized)

**P-1: Fix streaming content for TTFT measurement**
- GLM TT sometimes returns empty `delta.content` on long generations
- Blocks reliable TTFT benchmarking

**P0: True TP sharding for MoE experts** (required for 30 tok/s)
- Currently MoE experts run replicated — single-chip latency bottleneck
- Implement expert parallelism across the 1x8 mesh
- Expected: large multi-x latency reduction

**P1: Fuse remaining Q/KV projections**
- Complete the DeepSeek-style fused MLA weight approach
- Target `q_path` and `kv_cache_update` stage costs

**P2: L1/sharded memory for decode hot tensors**
- Move frequently-accessed tensors from DRAM to L1
- Enable fast kernel configs (`LoFi`, `packer_l1_acc`) where quality allows

**P3: MoE router + expert tuning for tiny decode batches**
- Sparse program config tuning
- Avoid row-major/tile layout churn

**P4: Deeper quantization (BF4 for MoE weights)**
- Only if still below target after P0-P3
- Must pass BF16-quality gates

**P5: Long benchmark matrix**
- Only after short-loop trending upward
- Run `repeat10`, `linear10`, `prefix5` suites

### 8.2 Functional Roadmap

- [ ] Native FlashMLA prefill (replace decode-loop fallback)
- [ ] Enable prefix caching (`supports_prefix_caching=True`)
- [ ] Increase `MAX_NUM_SEQS` beyond 1 (batched inference)
- [ ] 8-bit checkpoint ingestion (TT block-float conversion pipeline)
- [ ] Soak test (hours-long stability)
- [ ] OpenCode coding suite gate

---

## 9. Key Technical Decisions

### 9.1 Track A Only (TT vLLM Fork)

**Decision:** Stay on the Tenstorrent vLLM fork. Do not attempt upstream vLLM integration.

**Rationale:** Upstream integration is a separate multi-week project that would dilute
GLM bring-up focus and risk Qwen regressions. Keep TT vLLM changes small and model-gated.

### 9.2 Config Shim (Not Transformers 5)

**Decision:** Add `Glm4MoeLiteConfig` to vLLM's `_CONFIG_REGISTRY` rather than upgrading
to Transformers 5 which natively supports `glm4_moe_lite`.

**Rationale:** Transformers 5 is a major version bump. Risk of breaking Qwen and other
model support. The config shim is ~125 lines and fully sufficient.

### 9.3 MLA Default (Not Disabled)

**Decision:** Production path uses MLA (`use_mla=True`, KVPE cache `head_size=576`).
Debug fallback via `VLLM_MLA_DISABLE=1` available but not default.

**Rationale:** MLA is the correct production path — 20x smaller KV cache vs full K/V.
Non-MLA path is correctness-only and impractical for long context.

### 9.4 Sparse MoE (Not Dense)

**Decision:** Use `ttnn.sparse_matmul` for expert execution, not per-expert dense loops.

**Rationale:** Dense per-expert execution (`dense_decode`) was unstable on this stack
(container restarts). Sparse matmul is the production-supported path and handles
top-k dispatch efficiently.

### 9.5 Thinking Disabled by Default

**Decision:** GLM chat template patched to disable thinking by default.
Users can opt-in via `chat_template_kwargs: {"enable_thinking": true}`.

**Rationale:** Thinking mode produces long hidden reasoning that inflates latency
and confuses most UIs. For a ~6 tok/s endpoint, this is impractical. Disabling by
default reduced latency 2-3x.

### 9.6 BF16 Dense Weights (Not BF8)

**Decision:** Dense weights default to BF16. BF8 is opt-in via `GLM4_MOE_LITE_DENSE_TT_DTYPE`.

**Rationale:** BF8 dense trial was slower (1.91 vs 2.64 tok/s) and did not improve quality.
Expert weights use BF8 (smaller and tolerated by MoE math), but dense path stays BF16.

---

## 10. Risk Register

### High Risk

1. **MoE expert TP complexity** — Expert parallelism across mesh is non-trivial.
   DeepSeek's implementation provides a reference but GLM's routing differs.

2. **Prefill scalability** — Current decode-loop prefill has O(seq_len) host round-trips.
   Long prompts will have unacceptable TTFT until native FlashMLA prefill is implemented.

3. **Long-context stability** — Not tested beyond ~32k tokens. Memory behavior at
   scale is unknown. Paged cache works but has not been stress-tested.

### Medium Risk

4. **Streaming content bug** — Some long generations return empty deltas. Blocks
   reliable benchmarking and may affect production UX.

5. **8-bit conversion path** — TT uses block-float, not standard INT8/FP8.
   Conversion pipeline needs design and quality validation.

6. **Upstream vLLM drift** — The TT fork will diverge further from upstream.
   Backporting features (prefix caching improvements, scheduler updates) gets harder.

### Low Risk

7. **Qwen regression** — Mitigated by mandatory smoke tests after every change.
8. **Config/parsing gaps** — Mitigated by local config shim and model-gated code.

---

## 11. Reference Material

### 11.1 Canonical Planning Documents

Located at `/home/ttuser/src_docker/plan/glm47_flash/`:

| File | Contents |
|------|----------|
| `overall.md` | Master plan (697 lines) — phases, constraints, risk register, 7 self-critique iterations, execution updates |
| `resume.md` | Current status runbook — services, correctness gates, running commands, known issues |
| `perf-opt.md` | Performance iteration history (975 lines) — every benchmark run with measured tok/s |
| `migrate_model_to_tt.md` | Generic porting playbook (320 lines) — reusable for future models |
| `phase_1.md` through `phase_7.md` | Detailed per-phase plans |
| `checklist.md` | Quick pre-coding / pre-merge / pre-ship checklist |
| `baseline_qwen32b.md` | Qwen32B known-good baseline snapshot |

### 11.2 Appendices

| File | Contents |
|------|----------|
| `appendix_vllm_tt_contract.md` | Authoritative TT backend contract (471 lines) |
| `appendix_deep_diff_ttstack_vs_upstream_vllm.md` | Deep diff TT vs upstream vLLM (238 lines) |
| `appendix_tt_8bit_weights.md` | 8-bit weight conversion strategy (66 lines) |
| `appendix_deepseek_diff_glm4_moe_lite.md` | GLM vs DeepSeek TT implementation diff (137 lines) |
| `appendix_moe_inference_perf_lit_review.md` | MoE inference performance literature review (93 lines) |

### 11.3 Benchmark Artifacts

Located at `/home/ttuser/src_docker/plan/glm47_flash/artifacts/`:
- `benchmark_8087_vs_8088_*.md/.json` — A/B throughput snapshots
- `determinism_probe_*.md` — Determinism validation results
- `perf_iterations/iteration_*.md/.json` — All benchmark iteration results (30+ runs)

### 11.4 Environment Variables Reference

Over 30 environment variables control GLM behavior. Key categories:

**Model selection:** `HF_MODEL`, `MESH_DEVICE`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`

**Performance:** `GLM4_MOE_LITE_TP`, `GLM4_MOE_LITE_FUSE_QKV_A`, `GLM4_MOE_LITE_PREFILL_IMPL`,
`GLM4_MOE_LITE_MLA_USE_V_CACHE_SLICE`

**MoE routing:** `GLM4_MOE_LITE_MOE_ROUTER_IMPL` (tt/cpu), `GLM4_MOE_LITE_MOE_EXPERTS_IMPL` (sparse/dense_decode)

**Math fidelity:** `GLM4_MOE_LITE_MOE_SPARSE_FIDELITY`, `GLM4_MOE_LITE_MLA_FIDELITY`,
`GLM4_MOE_LITE_MOE_SPARSE_FP32_ACC`, `GLM4_MOE_LITE_MLA_FP32_ACC`

**Dtype:** `GLM4_MOE_LITE_DENSE_TT_DTYPE`, `GLM4_MOE_LITE_EXPERTS_TT_DTYPE`,
`GLM4_MOE_LITE_KV_CACHE_TT_DTYPE`

**Debug:** `GLM4_MOE_LITE_PROFILE`, `GLM4_MOE_LITE_MOE_SPARSE_DEBUG`,
`GLM4_MOE_LITE_DISABLE_DEALLOC`, `GLM4_MOE_LITE_DEBUG_PAGE_TABLE_BOUNDARY`

**UX:** `GLM_DEFAULT_DISABLE_THINKING`, `GLM_CHAT_TEMPLATE_FILE`

### 11.5 Git History Summary

**docker_tt** (14 GLM commits):
```
0255626 docs: document standard dev run in WORKLOG
56c779f dev: make perf-trace-tp the default GLM env
e968c60 dev: add DeepseekV3 registration, parameterized block-size
8e403bf docs: refresh GLM47 worklog SHAs
cfe8358 docs: add GLM47 Flash worklog pointer
38e6035 from_source: support fork repos + worktree-safe workspace base
6efd89f dev: GLM47 safe defaults + TT debug knobs
8a820a0 tools: add determinism probe for chat completions
11b9f10 glm47: correctness-first env defaults
35c1767 dev: enable GLM4_MOE_LITE_PACKER_L1_ACC in perf env
5bdaaa4 perf harness: parametrize remote GLM + health timeout
574054d docker_tt: ignore dev/.tmp.env.* files
70f1090 docker_tt: make TTNN_CONFIG_OVERRIDES safe + plumb loguru level
33eb401 glm47: add docker bring-up envs, benchmarks, smoke tests
```

**tt-metal** (10 GLM commits):
```
3b63e3cc34 glm4_moe_lite: avoid FlashMLA KV-boundary corruption
500f2e15f9 glm4_moe_lite: fix MoE decode corruption
d776d94cea glm4_moe_lite: env toggle packer_l1_acc for perf
ee73915cb8 glm4_moe_lite: block decode trace replay for correctness
5ad783dec8 glm4_moe_lite: fix dense_decode mesh expert slicing
b1fdf7c85f tt-metal: fix GLM decode trace RoPE host->device corruption
6209f449c5 glm4: mitigate TTNN view dealloc corruption
3d72cdc5ff glm4: fix KVPE cache shape in vLLM adapter
150995d339 glm4_moe_lite: vllm runner bring-up fixes and profiling
73864ced3c Add GLM4 MoE Lite demo scaffold
```

**vllm** (2 GLM commits):
```
b2fbf06a6 tt: add optional GLM page_table boundary logs
fe3586363 tt/vllm: GLM defaults, parsers, and config plumbing
```
