# Phase Checklist Template

Copy this into `plan/<short_name>/checklist.md` and track progress.

## Phase 1: Baselines and Guardrails
- [ ] Qwen 32B smoke test exists and passes
- [ ] Baseline runtime details recorded (SHAs, versions)
- [ ] Workspace created and documented
- [ ] Reference endpoint verified and reachable
- [ ] Model weights cached locally

## Phase 2: vLLM Enablement
- [ ] Config shim created (if model_type not in Transformers 4.x)
- [ ] Config registered in `_CONFIG_REGISTRY`
- [ ] Head size calculation correct:
  - Standard models: `hidden_size / num_attention_heads`
  - MLA models: `kv_lora_rank + qk_rope_head_dim` (MLA enabled)
  - MLA debug: `qk_nope_head_dim + qk_rope_head_dim` (MLA disabled)
- [ ] TT architecture registered in `platforms/tt.py`
- [ ] TT architecture registered in `entrypoint.sh`
- [ ] Config unit test passes
- [ ] Qwen 32B smoke test passes

## Phase 3: Skeleton + Integration
- [ ] tt-metal model directory created
- [ ] Weight loader works (lazy, bounded RAM)
- [ ] generator_vllm.py implements interface (placeholder forward ok)
- [ ] vLLM server starts and returns `/health` OK
- [ ] `/v1/models` shows the model
- [ ] `/v1/chat/completions` returns a response (even if nonsense)
- [ ] Qwen 32B smoke test passes

## Phase 4: Layer 0 Correctness (Prefill)
- [ ] Embedding matches reference (PCC >= 0.9999)
- [ ] RMSNorm matches reference (PCC >= 0.999)
- [ ] Attention projections match reference (PCC >= 0.999)
- [ ] RoPE matches reference (verify interleave/packing)
- [ ] MLA/attention output matches reference (PCC >= 0.99)
- [ ] Dense MLP matches reference (PCC >= 0.99)
- [ ] End-to-end layer 0 output matches reference
- [ ] Reference endpoint comparison: same prompt produces similar first token
- [ ] Qwen 32B smoke test passes

## Phase 5: KV Cache + Decode
- [ ] Paged KV cache fill works (prefill)
- [ ] Paged KV cache update works (decode)
- [ ] Single-step decode matches prefill-at-last-token reference
- [ ] Multi-step decode matches reference token-by-token (8+ steps)
- [ ] Variable-length batch decode works (no cross-contamination)
- [ ] No device hangs in repeated decode loops
- [ ] Reference endpoint comparison: first 16 tokens match
- [ ] Qwen 32B smoke test passes

## Phase 6: Full Model
- [ ] All layers run (attention + MLP/MoE for each layer)
- [ ] MoE routing matches reference (if applicable)
- [ ] Token-by-token match for 32 tokens on curated prompt suite (temp=0)
- [ ] No NaN/Inf in any output
- [ ] Soak test: 100+ requests without hang or crash
- [ ] No monotonic memory growth
- [ ] Reference endpoint comparison: responses are qualitatively similar
- [ ] First A/B benchmark artifact saved
- [ ] Qwen 32B smoke test passes

## Phase 7: Productionization
- [ ] Manual validation: /health, /v1/models, short prompt, medium prompt
- [ ] Manual validation: tool-calling works (if applicable)
- [ ] Manual validation: Open WebUI works on :3000
- [ ] Performance target defined and measured
- [ ] Stage-level profiling done (identify bottlenecks)
- [ ] Startup time measured and documented
- [ ] Default configuration documented
- [ ] Smoke test passes reliably
- [ ] A/B benchmark vs reference is stable and documented
- [ ] Qwen 32B smoke test passes
- [ ] Ready for upstream PR slices
