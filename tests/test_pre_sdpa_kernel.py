#!/usr/bin/env python3
"""Standalone test for PreSDPA fused kernel debug_stage binary search.

Run inside the container:
    docker exec dev-vllm-tt-1 python /workspace/docker_tt/tests/test_pre_sdpa_kernel.py --stage 0

This avoids warmup/flash MLA L1 conflicts by testing the kernel in isolation.
"""
import argparse
import os
import sys
import time

# Must set before importing ttnn
os.environ.setdefault("TT_METAL_HOME", "/tt-metal")
os.environ.setdefault("LOGURU_LEVEL", "INFO")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0,
                        help="debug_stage: 0=noop, 1=rmsnorm, 2=mcast, ... 99=all")
    parser.add_argument("--layer", type=int, default=0,
                        help="Which layer's weights to use")
    args = parser.parse_args()

    print(f"[TEST] Starting PreSDPA kernel test with debug_stage={args.stage}")

    import ttnn
    from pathlib import Path

    # Open mesh device
    print("[TEST] Opening mesh device...")
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 8),
        dispatch_core_config=ttnn.DispatchCoreConfig(
            ttnn.DispatchCoreType.WORKER
        ),
    )
    mesh.enable_program_cache()
    print(f"[TEST] Mesh device opened: {mesh.num_devices()} devices")

    # Load model config
    from models.demos.glm4_moe_lite.tt.layer_weights import (
        _prepare_fused_pre_sdpa_weights,
        LazyStateDict,
    )
    from models.demos.glm4_moe_lite.tt.model_tt import Glm4MoeLiteHParams

    snapshot_dir = "/cache/huggingface/hub/models--zai-org--GLM-4.7-Flash/snapshots/7dd20894a642a0aa287e9827cb1a1f7f91386b67"
    cache_dir = Path("/root/.cache/ttnn/models/glm4_moe_lite/vllm")

    # Load HF config
    import json
    with open(os.path.join(snapshot_dir, "config.json")) as f:
        hf_config = json.load(f)

    hparams = Glm4MoeLiteHParams(
        hidden_size=hf_config["hidden_size"],
        num_attention_heads=hf_config["num_attention_heads"],
        q_lora_rank=hf_config["q_lora_rank"],
        kv_lora_rank=hf_config["kv_lora_rank"],
        qk_nope_head_dim=hf_config["qk_nope_head_dim"],
        qk_rope_head_dim=hf_config["qk_rope_head_dim"],
        v_head_dim=hf_config["v_head_dim"],
        rms_norm_eps=hf_config["rms_norm_eps"],
        num_hidden_layers=hf_config["num_hidden_layers"],
        intermediate_size=hf_config.get("intermediate_size", 0),
        num_key_value_heads=hf_config.get("num_key_value_heads", hf_config["num_attention_heads"]),
        vocab_size=hf_config.get("vocab_size", 0),
        first_k_dense_replace=hf_config.get("first_k_dense_replace", 0),
        num_experts_per_tok=hf_config.get("num_experts_per_tok", 0),
        n_routed_experts=hf_config.get("n_routed_experts", 0),
        moe_intermediate_size=hf_config.get("moe_intermediate_size", 0),
        n_shared_experts=hf_config.get("n_shared_experts", 0),
        shared_expert_intermediate_size=hf_config.get("shared_expert_intermediate_size", 0),
        norm_topk_prob=hf_config.get("norm_topk_prob", True),
    )

    # Create lazy state dict
    state = LazyStateDict(snapshot_dir)

    # Prepare fused weights
    print(f"[TEST] Preparing fused pre-SDPA weights for layer {args.layer}...")
    t0 = time.monotonic()
    fps = _prepare_fused_pre_sdpa_weights(
        device=mesh,
        state=state,
        layer_idx=args.layer,
        hparams=hparams,
        cache_dir=cache_dir,
        dense_dtype=ttnn.bfloat8_b,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    t1 = time.monotonic()
    print(f"[TEST] Fused weights prepared in {t1 - t0:.1f}s")
    print(f"[TEST] Weight keys: {list(fps.keys())}")

    # Prepare input tensor (dummy x for decode: [1, hidden_size])
    import torch
    hidden = int(hparams.hidden_size)
    x_torch = torch.randn(1, hidden, dtype=torch.bfloat16)
    x_for_fused = ttnn.from_torch(
        x_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=fps["input_mem_config"],
        tile=ttnn.Tile((1, 32)),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    print(f"[TEST] Input tensor created: {x_for_fused.shape}")

    # Run the kernel
    from models.demos.glm4_moe_lite.fused_ops.pre_sdpa.op import PreSDPA

    print(f"[TEST] Calling PreSDPA.op() with debug_stage={args.stage}...")
    t0 = time.monotonic()
    sdpa_output = PreSDPA.op(
        x_for_fused,
        fps["intermediate_tensor"],
        fps["gamma"],
        fps["matmul_weights"],
        fps.get("rmsnorm2_gamma"),
        fps.get("matmul2_weights"),
        fps.get("matmul3_weights"),
        fps.get("cos"),
        fps.get("sin"),
        fps.get("trans_mat"),
        fps.get("dkv_matmul_weights"),
        fps.get("dkv_rmsnorm_gamma"),
        fps["output_tensor"],
        fps["sender_coord"],
        semaphores=None,
        cluster_axis=0,
        secondary_cluster_axis=1,
        using_persistent_buffers=True,
        epsilon=fps["epsilon"],
        fp32_dest_acc_en=True,
        skip_ccl=True,
        debug_stage=args.stage,
    )
    t1 = time.monotonic()
    print(f"[TEST] PreSDPA.op() returned in {(t1 - t0)*1000:.1f}ms")

    # Sync device to check for hangs
    print("[TEST] Syncing device...")
    t0 = time.monotonic()
    ttnn.synchronize_device(mesh)
    t1 = time.monotonic()
    print(f"[TEST] Device sync OK in {(t1 - t0)*1000:.1f}ms")

    # Deallocate everything
    print("[TEST] Deallocating...")
    ttnn.deallocate(sdpa_output, force=True)
    ttnn.deallocate(x_for_fused, force=True)
    for k, v in list(fps.items()):
        if hasattr(v, "is_allocated") and callable(v.is_allocated):
            try:
                ttnn.deallocate(v, force=True)
            except Exception:
                pass

    print(f"[TEST] SUCCESS: debug_stage={args.stage} completed without hang!")
    print("[TEST] Closing mesh device...")
    ttnn.close_mesh_device(mesh)
    print("[TEST] Done.")

if __name__ == "__main__":
    main()
