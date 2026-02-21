#!/usr/bin/env python3
"""DRAM bandwidth diagnostic: interleaved vs DRAM-sharded matmul on WH N300.

Measures effective DRAM bandwidth for decode-sized matmuls (M=32) comparing:
  (a) DRAM-INTERLEAVED weights (current GLM-4.7-Flash default)
  (b) DRAM-SHARDED weights (saturating DRAM bandwidth technique)

Uses representative weight shapes from GLM-4.7-Flash:
  - w_o:    K=5120, N=2048  (attention output projection)
  - w_gate: K=2048, N=10240 (dense MLP gate)
  - w_down: K=10240, N=2048 (dense MLP down)
  - w_q_a:  K=2048, N=768   (attention q compress)
"""

import sys
import time

import torch
import ttnn

# Import DRAM-sharded helpers from DeepSeek V3
sys.path.insert(0, "/home/ttuser/src_docker/ws/glm47_flash_small_wormhole/tt-metal")
from models.demos.deepseek_v3.utils.config_helpers import (
    dram_sharded_weight_config,
    get_activation_sharding_core_counts_for_dram_matmul,
    get_dram_sharded_matmul_config,
)


def benchmark_interleaved_matmul(device, M, K, N, weight_tt, num_iters=1000):
    """Benchmark matmul with DRAM-interleaved weight."""
    # Create activation
    act_torch = torch.randn(1, 1, M, K).bfloat16().float()
    act_tt = ttnn.from_torch(
        act_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Warmup
    for _ in range(10):
        out = ttnn.linear(act_tt, weight_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out, force=True)

    ttnn.synchronize_device(device)

    # Timed run
    t0 = time.perf_counter()
    for _ in range(num_iters):
        out = ttnn.linear(act_tt, weight_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out, force=True)
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - t0

    ttnn.deallocate(act_tt, force=True)
    return elapsed


def benchmark_dram_sharded_matmul(device, M, K, N, weight_tt, num_iters=1000):
    """Benchmark matmul with DRAM-sharded weight."""
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y

    input_cores = max(get_activation_sharding_core_counts_for_dram_matmul(K, max_cores))
    output_cores = max(get_activation_sharding_core_counts_for_dram_matmul(N, max_cores))

    prog_cfg = get_dram_sharded_matmul_config(
        m=M, k=K, n=N,
        input_num_shards=input_cores,
        output_num_shards=output_cores,
    )

    # Activation L1 width-sharded config
    act_shard_shape = (
        ttnn.core.roundup(M, ttnn.TILE_SIZE),
        ttnn.core.roundup(K // input_cores, ttnn.TILE_SIZE),
    )
    act_mem_config = ttnn.create_sharded_memory_config_(
        shape=act_shard_shape,
        core_grid=ttnn.num_cores_to_corerangeset(
            input_cores,
            ttnn.CoreCoord(grid.x, grid.y),
            row_wise=True,
        ),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    # Create activation (interleaved first, then shard to L1)
    act_torch = torch.randn(1, 1, M, K).bfloat16().float()
    act_tt = ttnn.from_torch(
        act_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    act_sharded = ttnn.to_memory_config(act_tt, act_mem_config)
    ttnn.deallocate(act_tt, force=True)

    # Warmup
    for _ in range(10):
        out = ttnn.linear(
            act_sharded, weight_tt,
            program_config=prog_cfg,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )
        ttnn.deallocate(out, force=True)

    ttnn.synchronize_device(device)

    # Timed run
    t0 = time.perf_counter()
    for _ in range(num_iters):
        out = ttnn.linear(
            act_sharded, weight_tt,
            program_config=prog_cfg,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )
        ttnn.deallocate(out, force=True)
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - t0

    ttnn.deallocate(act_sharded, force=True)
    return elapsed


def run_comparison(device, name, M, K, N, num_iters=1000):
    """Run interleaved vs DRAM-sharded comparison for one matmul shape."""
    print(f"\n{'='*70}")
    print(f"  {name}: M={M}, K={K}, N={N}")
    print(f"  Weight size: {K*N*2 / 1e6:.2f} MB (BF16)")
    print(f"{'='*70}")

    dram_grid_size = device.dram_grid_size()
    weight_bytes = K * N * 2  # BF16

    # Create weight tensors
    w_torch = torch.randn(1, 1, K, N).bfloat16().float()

    # --- DRAM-INTERLEAVED weight ---
    w_interleaved = ttnn.from_torch(
        w_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    elapsed_interleaved = benchmark_interleaved_matmul(device, M, K, N, w_interleaved, num_iters)
    ttnn.deallocate(w_interleaved, force=True)

    bw_interleaved = (weight_bytes * num_iters) / elapsed_interleaved / 1e9

    print(f"  INTERLEAVED: {elapsed_interleaved*1000:.1f} ms / {num_iters} iters"
          f"  = {elapsed_interleaved/num_iters*1e6:.1f} us/iter"
          f"  BW = {bw_interleaved:.1f} GB/s")

    # --- DRAM-SHARDED weight ---
    w_sharded_mem_config = dram_sharded_weight_config(K, N, dram_grid_size)
    w_sharded = ttnn.from_torch(
        w_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=w_sharded_mem_config,
    )

    elapsed_sharded = benchmark_dram_sharded_matmul(device, M, K, N, w_sharded, num_iters)
    ttnn.deallocate(w_sharded, force=True)

    bw_sharded = (weight_bytes * num_iters) / elapsed_sharded / 1e9

    print(f"  DRAM-SHARDED: {elapsed_sharded*1000:.1f} ms / {num_iters} iters"
          f"  = {elapsed_sharded/num_iters*1e6:.1f} us/iter"
          f"  BW = {bw_sharded:.1f} GB/s")

    ratio = bw_sharded / bw_interleaved if bw_interleaved > 0 else float('inf')
    print(f"  SPEEDUP: {ratio:.1f}x bandwidth improvement")
    print(f"  (WH spec: 288 GB/s @ 12 Gbps, 336 GB/s @ 14 Gbps)")

    return {
        "name": name,
        "K": K, "N": N,
        "bw_interleaved": bw_interleaved,
        "bw_sharded": bw_sharded,
        "ratio": ratio,
        "us_interleaved": elapsed_interleaved / num_iters * 1e6,
        "us_sharded": elapsed_sharded / num_iters * 1e6,
    }


def main():
    M = 32  # Decode batch (padded to tile)

    # GLM-4.7-Flash representative matmul shapes
    test_cases = [
        ("w_o (attn output)",   5120, 2048),   # 20 heads * 256 v_head_dim -> hidden
        ("w_gate (dense MLP)",  2048, 10240),   # hidden -> intermediate
        ("w_down (dense MLP)",  10240, 2048),   # intermediate -> hidden
        ("w_q_a (q compress)",  2048, 768),     # hidden -> q_lora_rank
        ("w_kv_a (kv compress)", 2048, 576),    # hidden -> kv_lora_rank + rope_dim (512+64)
    ]

    # Open device
    device = ttnn.open_device(device_id=0)
    print(f"Device: {device}")
    print(f"Compute grid: {device.compute_with_storage_grid_size()}")
    print(f"DRAM grid: {device.dram_grid_size()}")

    results = []
    for name, K, N in test_cases:
        try:
            r = run_comparison(device, name, M, K, N, num_iters=1000)
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Name':<25} {'Interleaved':>12} {'Sharded':>12} {'Ratio':>8}")
    print(f"  {'':.<25} {'(GB/s)':>12} {'(GB/s)':>12} {'':>8}")
    for r in results:
        print(f"  {r['name']:<25} {r['bw_interleaved']:>10.1f}   {r['bw_sharded']:>10.1f}   {r['ratio']:>6.1f}x")

    print(f"\n  Per-iteration latency:")
    print(f"  {'Name':<25} {'Interleaved':>12} {'Sharded':>12} {'Speedup':>8}")
    print(f"  {'':.<25} {'(us)':>12} {'(us)':>12} {'':>8}")
    for r in results:
        speedup = r['us_interleaved'] / r['us_sharded'] if r['us_sharded'] > 0 else float('inf')
        print(f"  {r['name']:<25} {r['us_interleaved']:>10.1f}   {r['us_sharded']:>10.1f}   {speedup:>6.1f}x")

    ttnn.close_device(device)
    print("\nDone.")


if __name__ == "__main__":
    main()
