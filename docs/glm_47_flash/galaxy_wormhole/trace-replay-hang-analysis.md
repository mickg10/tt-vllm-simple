# Galaxy TG Mesh Trace Replay Hang — Root Cause Analysis

**Date**: 2026-02-22
**Status**: Root cause confirmed, Option A fix REJECTED (introduces new deadlock), Option B (device barrier) recommended but not yet implemented
**Affected**: Galaxy 32-chip TG mesh (MESH_DEVICE=TG, 8x4 grid)
**Not affected**: T3K 8-chip mesh

## Bug Summary

`ttnn.execute_trace(blocking=True)` hangs after ~100-150 cumulative decode iterations on Galaxy TG mesh. Non-traced decode is stable for 1000+ tokens, confirming the bug is 100% trace-specific.

## Root Cause: Host-Device State Synchronization Race Condition

### The Core Problem

`update_worker_state_post_trace_execution()` in `tt_metal/impl/trace/dispatch.cpp` **unconditionally advances host-side state** (write pointers, expected worker counts) after every `enqueue_trace` call, **before confirming all 32 devices completed**.

### Critical Code Path

```
enqueue_trace()
  ├── for each of 32 devices: issue_trace_commands()   // async dispatch
  ├── reset_prefetcher_cache_manager()
  ├── update_worker_state_post_trace_execution()        // HOST STATE UPDATED HERE
  │     ├── expected_num_workers_completed = desc.num_completion_worker_cores  // UNCONDITIONAL
  │     ├── set_mcast_wptr(N_programs & 7)              // RING BUFFER ADVANCED
  │     └── mark_completely_full(expected_workers)
  └── finish_nolock()                                   // THEN waits for completion
        ├── enqueue_record_event_to_host_nolock()
        └── wait(num_outstanding_reads_ == 0)           // Race: event ≠ true device completion
```

### The Race Condition (Detailed)

1. **Iteration N**: Host issues `enqueue_trace` to all 32 devices. Some devices lag due to fabric contention, CCL synchronization, or thermal variation.

2. **finish_nolock returns**: The host processes completion events from the command queue. But event processing on the host is not perfectly synchronized with the device's absolute final completion. There is a tiny window where the host believes all devices are done while a slow device (e.g., device 29) hasn't fully finished.

3. **Iteration N+1**: Host calls `enqueue_trace` again. `update_worker_state_post_trace_execution` resets the write pointer and expected worker counts. The go signal for N+1 is issued with `expected_num_workers_completed` from N's post-trace state.

4. **Device lags accumulate**: Over ~100-150 iterations, the mismatch between host-side state and actual device state grows. The `add_dispatch_go_signal_mcast` command includes `expected_num_workers_completed[index]` which tells the dispatch core how many workers should have completed before sending the next go signal.

5. **Deadlock**: Eventually, a device receives a go signal with an `expected_num_workers_completed` value that doesn't match its actual completion count. The dispatch core stalls waiting for workers that are in an inconsistent state. Because the model uses all_reduce (CCL), one stalled device blocks its entire TP=8 replica, and all subsequent requests hang.

### Why T3K Works

- Only 8 devices (vs 32): 4x lower probability of any device lagging
- Tighter synchronization: fewer fabric hops, less contention
- The latent bug exists but the timing window is too small to trigger at 8-device scale

### Why Non-Traced Mode Works

- Eager mode doesn't use `update_worker_state_post_trace_execution`
- Each operation is individually dispatched with fresh state
- No accumulated state drift across iterations

### Key Constants

- `launch_msg_buffer_num_entries = 8` (ring buffer depth)
- `set_mcast_wptr(val)` uses `val & 7` (wraps to 0-7)
- GLM-4.7-Flash trace has ~660 programs → `660 & 7 = 4` (wptr always lands at 4)
- The wptr itself doesn't overflow, but the `expected_num_workers_completed` tracking gets stale

## Affected Source Files

| File | Lines | Role |
|------|-------|------|
| `tt_metal/impl/trace/dispatch.cpp` | 194-220 | `update_worker_state_post_trace_execution` — unconditional host state update |
| `tt_metal/impl/trace/dispatch.cpp` | 79-167 | `issue_trace_commands` — go signal with `expected_num_workers_completed` |
| `tt_metal/distributed/fd_mesh_command_queue.cpp` | 1109-1150 | `enqueue_trace` — sequential dispatch to 32 devices |
| `tt_metal/distributed/fd_mesh_command_queue.cpp` | 533-561 | `finish_nolock` — event-based completion wait (may return early) |
| `tt_metal/impl/dispatch/launch_message_ring_buffer_state.cpp` | 23-28 | `set_mcast_wptr` — ring buffer write pointer (8 entries) |
| `tt_metal/hw/inc/hostdev/dev_msgs.h` | 378 | `launch_msg_buffer_num_entries = 8` |

## Proposed Fixes

### Option A: Move state update after finish — TESTED, DOES NOT WORK (2026-02-24)

In `FDMeshCommandQueue::enqueue_trace()`, move `update_worker_state_post_trace_execution` AFTER `finish_nolock`.

**RESULT: Introduces NEW deadlock on first trace replay during warmup.** The `finish_nolock()` call
and `update_worker_state_post_trace_execution()` have an implicit dependency — `finish_nolock()` needs
the state from `update_worker_state_post_trace_execution()` to correctly complete its event wait.
Moving the wait before the state update leaves the system with stale worker state, causing the
dispatch core to wait for a completion condition that never triggers.

The initial "successful" test (Run 5) was a false positive — the build cache prevented the fix from
actually being compiled. All successful Galaxy benchmarks (Runs 5, 6) used the ORIGINAL unfixed code.

**This option is REJECTED.** Do not attempt this reorder.

### Option B: Add device-level completion barrier (RECOMMENDED)

After `finish_nolock`, add a polling loop that reads a completion flag from each of the 32 devices:

```cpp
if (blocking) {
    this->finish_nolock();
    // Additional barrier: verify each device is truly done
    for (auto* device : mesh_device_->get_devices()) {
        device->wait_for_trace_completion(trace_id);
    }
}
```

**Risk**: Performance impact from per-device polling. This is now the recommended approach since
Option A was shown to introduce a worse deadlock. The per-device polling would add ~32 * barrier_cost
per trace replay, but at ~15us per device this is negligible vs the 115ms ITL.

**Note**: `wait_for_trace_completion()` does not exist in the current API. It would need to be
implemented, possibly using the existing `finish()` infrastructure but on a per-device basis.

### Option C: Increase ring buffer size (Mitigation, not fix)

Change `launch_msg_buffer_num_entries` from 8 to a larger power of 2 (e.g., 64 or 128). This would increase the tolerance for state drift but not eliminate the root cause.

## Reproducer

```bash
# On Galaxy (32-chip TG mesh):
ssh -p 55211 user@38.97.6.6
# Reset devices
sudo tt-smi -glx_reset_auto
sudo rm -f /dev/shm/TT_UMD_LOCK.*
# Start container with trace_mode=decode_only
cd /home/user/src_docker/docker_tt
sg docker -c "docker compose --env-file dev/.env.glm47.galaxy -f dev/docker-compose.yml -f dev/docker-compose.galaxy.yml up -d --force-recreate vllm-tt"
# Wait for healthy (~90s)
# Send request with gen>200:
curl http://localhost:8088/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"zai-org/GLM-4.7-Flash","prompt":"Hello","max_tokens":300,"temperature":0.7}'
# Will hang after ~100-150 decode tokens
```

## Validation Plan (Updated 2026-02-24)

Option A was tested and FAILED. Next steps:

1. Design Option B implementation (per-device completion barrier)
2. Implement `wait_for_trace_completion()` or equivalent per-device sync
3. Rebuild and test on Galaxy with gen=500 bs=1 — should complete without hang
4. Test gen=500 bs=32 — should complete without hang
5. Run 10x sequential gen=500 requests — should all complete
6. Verify T3K still works (regression test)

### Build Procedure (for future fix attempts)

```bash
# Inside the container:
cd /tt-metal
# Edit the source file
vi tt_metal/distributed/fd_mesh_command_queue.cpp
# Touch to force cmake detection
touch tt_metal/distributed/fd_mesh_command_queue.cpp
# Build only the tt_metal target (fast, ~30s)
cmake --build build_Release --target tt_metal -j 32
# Copy to the runtime lib directory
cp build_Release/tt_metal/libtt_metal.so build_Release/lib/libtt_metal.so
# Verify
md5sum build_Release/lib/libtt_metal.so
ls -la build_Release/lib/libtt_metal.so
```

Then stop container, reset devices, start container.

## References

- `plan/glm47_flash/galaxy_wormhole/perf-opt.md` — Benchmark results showing the hang
- `plan/glm47_flash/galaxy_wormhole/galaxy-system-state.md` — System state during hang
- `tests/ttnn/unit_tests/base_functionality/test_multi_device_trace_TG.py` — TG trace test (only 15 iter, no CCL, passes)
