# Galaxy System State Snapshot

**Timestamp**: 2026-02-22 00:31 UTC
**Machine**: g15glx03 (Galaxy Wormhole, 32 chips)
**SSH**: `ssh -p 55211 user@38.97.6.6`

---

## 1. Device Health (32 Wormhole Chips)

**Summary**: ALL 32 devices HEALTHY. No faults, no errors, temperatures nominal.

| Dev# | Bus ID        | Board Type       | Temp (C) | Power (W) | AICLK | Voltage | DRAM | Faults |
|------|---------------|------------------|----------|-----------|-------|---------|------|--------|
| 0    | 0000:01:00.0  | tt-galaxy-wh L   | 38.1     | 35.0      | 1000  | 0.92    | OK   | 0x0    |
| 1    | 0000:02:00.0  | tt-galaxy-wh L   | 38.2     | 34.0      | 1000  | 0.93    | OK   | 0x0    |
| 2    | 0000:03:00.0  | tt-galaxy-wh L   | 37.6     | 32.0      | 1000  | 0.91    | OK   | 0x0    |
| 3    | 0000:04:00.0  | tt-galaxy-wh L   | 40.1     | 35.0      | 1000  | 0.91    | OK   | 0x0    |
| 4    | 0000:05:00.0  | tt-galaxy-wh L   | 37.3     | 36.0      | 1000  | 0.92    | OK   | 0x0    |
| 5    | 0000:06:00.0  | tt-galaxy-wh L   | 37.0     | 34.0      | 1000  | 0.91    | OK   | 0x0    |
| 6    | 0000:07:00.0  | tt-galaxy-wh L   | 38.1     | 34.0      | 1000  | 0.92    | OK   | 0x0    |
| 7    | 0000:08:00.0  | tt-galaxy-wh L   | 39.2     | 35.0      | 1000  | 0.91    | OK   | 0x0    |
| 8    | 0000:41:00.0  | tt-galaxy-wh L   | 38.4     | 35.0      | 1000  | 0.92    | OK   | 0x0    |
| 9    | 0000:42:00.0  | tt-galaxy-wh L   | 37.8     | 35.0      | 1000  | 0.92    | OK   | 0x0    |
| 10   | 0000:43:00.0  | tt-galaxy-wh L   | 38.6     | 35.0      | 1000  | 0.94    | OK   | 0x0    |
| 11   | 0000:44:00.0  | tt-galaxy-wh L   | 39.6     | 35.0      | 1000  | 0.90    | OK   | 0x0    |
| 12   | 0000:45:00.0  | tt-galaxy-wh L   | 37.9     | 34.0      | 1000  | 0.92    | OK   | 0x0    |
| 13   | 0000:46:00.0  | tt-galaxy-wh L   | 37.1     | 35.0      | 1000  | 0.91    | OK   | 0x0    |
| 14   | 0000:47:00.0  | tt-galaxy-wh L   | 37.1     | 35.0      | 1000  | 0.92    | OK   | 0x0    |
| 15   | 0000:48:00.0  | tt-galaxy-wh L   | 39.3     | 35.0      | 1000  | 0.92    | OK   | 0x0    |
| 16   | 0000:81:00.0  | tt-galaxy-wh L   | 37.9     | 35.0      | 1000  | 0.90    | OK   | 0x0    |
| 17   | 0000:82:00.0  | tt-galaxy-wh L   | 36.6     | 33.0      | 1000  | 0.88    | OK   | 0x0    |
| 18   | 0000:83:00.0  | tt-galaxy-wh L   | 37.9     | 34.0      | 1000  | 0.89    | OK   | 0x0    |
| 19   | 0000:84:00.0  | tt-galaxy-wh L   | 39.3     | 37.0      | 1000  | 0.90    | OK   | 0x0    |
| 20   | 0000:85:00.0  | tt-galaxy-wh L   | 37.1     | 35.0      | 1000  | 0.91    | OK   | 0x0    |
| 21   | 0000:86:00.0  | tt-galaxy-wh L   | 37.8     | 34.0      | 1000  | 0.88    | OK   | 0x0    |
| 22   | 0000:87:00.0  | tt-galaxy-wh L   | 37.6     | 35.0      | 1000  | 0.87    | OK   | 0x0    |
| 23   | 0000:88:00.0  | tt-galaxy-wh L   | 39.6     | 37.0      | 1000  | 0.91    | OK   | 0x0    |
| 24   | 0000:c1:00.0  | tt-galaxy-wh L   | 37.9     | 34.0      | 1000  | 0.90    | OK   | 0x0    |
| 25   | 0000:c2:00.0  | tt-galaxy-wh L   | 37.8     | 35.0      | 1000  | 0.91    | OK   | 0x0    |
| 26   | 0000:c3:00.0  | tt-galaxy-wh L   | 38.1     | 35.0      | 1000  | 0.90    | OK   | 0x0    |
| 27   | 0000:c4:00.0  | tt-galaxy-wh L   | 39.3     | 33.0      | 1000  | 0.88    | OK   | 0x0    |
| 28   | 0000:c5:00.0  | tt-galaxy-wh L   | 37.2     | 34.0      | 1000  | 0.90    | OK   | 0x0    |
| 29   | 0000:c6:00.0  | tt-galaxy-wh L   | 37.2     | 36.0      | 1000  | 0.91    | OK   | 0x0    |
| 30   | 0000:c7:00.0  | tt-galaxy-wh L   | 37.3     | 32.0      | 1000  | 0.89    | OK   | 0x0    |
| 31   | 0000:c8:00.0  | tt-galaxy-wh L   | 38.6     | 35.0      | 1000  | 0.88    | OK   | 0x0    |

**Temperature range**: 36.6 - 40.1 C (all well within safe limits)
**Power range**: 32.0 - 37.0 W per chip, ~1,104 W total
**All AICLK**: 1000 MHz (nominal)
**All DRAM**: OK
**All Faults**: 0x0 (none)
**ETH_LIVE_STATUS**: 0xffff (all ethernet links live)

**Driver/Firmware**:
- TT-KMD: 2.7.1-pre
- tt-smi: 4.0.0, pyluwen: 0.8.1, tt-umd: 0.9.2
- CM FW: 2.36.0.0 (dated 2025-10-01)
- ETH FW: 7.2.0
- FW Bundle: 19.1.0.0

---

## 2. Host System Resources

### Memory
```
              total        used        free      shared  buff/cache   available
Mem:          566Gi       143Gi       255Gi        30Mi       167Gi       417Gi
Swap:            0B          0B          0B
```
**Analysis**: 417 GiB available. No swap configured. Memory is healthy with 74% available.

### CPU / Load
```
00:31:11 up 10 days, 16:46,  0 users,  load average: 3.42, 3.57, 3.35
CPUs: 64
```
**Analysis**: Uptime 10 days. Load average ~3.5 on 64 cores = very light load (~5.5% utilization). Normal for inference serving (device-bound workload).

### Disk
```
Filesystem                         Size  Used Avail Use% Mounted on
/dev/mapper/ubuntu--vg-ubuntu--lv  894G   23G  868G   3% /
/dev/mapper/data--vg-data--lv      7.0T  143G  6.9T   2% /home
```
**Analysis**: Plenty of disk space. Root 3% used, /home 2% used (7 TB total).

### IO Stats
- NVMe disks: ~90 w/s at 1.4 MB/s on nvme4n1/nvme5n1, ~1.5% utilization
- Very light I/O. No bottleneck.

### Top Processes by RSS
| Process | RSS | Notes |
|---------|-----|-------|
| vLLM worker (spawn_main) | ~117 GB | Model weights in memory (expected for 32-chip GLM-4.7-Flash) |
| vLLM server (python) | ~6.4 GB | Server process |
| tt-smi (PID 82372) | ~580 MB | Running since Feb 17, **99% CPU** |
| tt-smi (PID 588319) | ~580 MB | Running since Feb 21, **99% CPU** |

**CONCERN**: Two tt-smi instances running at 99% CPU each. PID 82372 has been running since Feb 17 (5 days) and has consumed 6,331 minutes of CPU time. PID 588319 since Feb 21 with 476 minutes. These are likely orphaned monitoring processes. They waste ~2 cores continuously but are not harmful to inference performance.

---

## 3. Docker State

### Container Status
```
CONTAINER ID   NAME            STATUS                 PORTS
87f83109ab7b   dev-vllm-tt-1   Up 3 hours (healthy)   0.0.0.0:8088->8088/tcp
98ecca8c3289   dev-tt-monitor-1  Up 3 days             0.0.0.0:9090->9090/tcp
```

**Container Health**: `healthy`
**Restart Count**: **104** (significant - matches prior crash history, Task #1)
**Created**: 5 hours ago, Up 3 hours (suggests ~2 hours of crash/restart cycles before stabilizing)

### Container Resources
```
CPU %: 165.88%    MEM: 68.38 GiB / 566.1 GiB (12.08%)    NET I/O: 7.13MB / 71.7MB    PIDS: 503
```
**Analysis**: Container using ~1.7 cores CPU, 68 GiB memory, 503 PIDs. Normal for serving.

### Container Processes (inside)
1. **Main vLLM server** (PID 766853): Python vLLM api_server process
2. **Resource tracker** (PID 767874): multiprocessing resource tracker
3. **Model worker** (PID 767875): spawn_main worker at **77% CPU** (device operations)

---

## 4. Shared Memory and UMD State

### /dev/shm Contents
- **29 shared memory segments** (sm_segment.*), each 16 MB = ~464 MB total
- Segments from multiple container IDs (stale segments from prior containers: `0e771f`, `122329`, `4c767d`, `8667e2`, `b274ed`, `c63257`, `e28ef7`, `f0d78d`, `f13218`)
- Active container segments: `87f83109ab7b` (7 segments, ~112 MB)
- **No TT_UMD_LOCK.* files** present (good - no stale locks)

### /dev/shm Size
```
tmpfs   284G  3.6M  284G   1% /dev/shm
```
**Analysis**: Only 3.6 MB reported as used (filesystem metadata), despite ~464 MB of segment files. Plenty of headroom.

**NOTE**: There are stale shared memory segments from 8+ previous container incarnations. These are harmless but could be cleaned up. They are NOT UMD locks, just multiprocessing shared memory buffers.

---

## 5. Network/Fabric

### dmesg
```
Found a Tenstorrent Wormhole device at bus 0000:43.
tenstorrent 0000:43:00.0: enabling device (0000 -> 0002)
[... repeated for 16 devices total in the last 30 lines ...]
```
**Analysis**: Only device enumeration messages. **No PCI errors, no AER errors, no fabric errors.** All clean.

### D-State Processes
**NONE** - No processes in uninterruptible sleep. All devices responsive.

---

## 6. Container Logs (Recent)

**Last 200 lines**: All routine `/metrics` and `/health` polling from the monitoring container (172.18.0.3) every ~10 seconds. Health checks returning 200 OK consistently.

**No errors, no warnings, no unusual log entries in the tail.**

### Startup Log (Key Events)
The container had a failed startup attempt at 21:39:14 with:
```
RuntimeError: No existing board type for board id 0xffffffffffffffff
```
This is a known transient error when devices are not yet fully initialized after a restart cycle. The container recovered on subsequent restart:

1. **21:40:54**: vLLM server started successfully
2. **21:41:26**: TTModelRunner initialized (trace_mode=decode_only, warmup=True)
3. **21:41:41**: Prefill warmup started
4. **21:42:40**: Prefill warmup finished (59 seconds)
5. **21:42:40**: Decode warmup started
6. **21:42:47**: Decode warmup finished (7 seconds)
7. **21:42:47**: Engine init took **81.03 seconds** total
8. **21:42:48**: Application startup complete, serving on port 8088

---

## 7. vLLM Serving State

### Health
```
Status: healthy (HTTP 200)
```

### Model
```json
{
  "id": "zai-org/GLM-4.7-Flash",
  "max_model_len": 32768,
  "owned_by": "vllm"
}
```

### Active Benchmark
A `bench_decode.py` process is currently running:
```
python3 tests/bench_decode.py --url http://localhost:8088 --gen-tokens 500 --only-batch 1 --skip-combined --prefill-contexts 0
```
Started at 00:09, running batch size 1 decode benchmark with 500 generation tokens.

### Metrics (from completed warmup request + active benchmark)
- **Requests running**: 1 (the active benchmark request)
- **Requests waiting**: 0
- **KV Cache usage**: 0.37% (nearly empty - single request)
- **Completed requests**: 1 (finished with reason=length, 20 gen tokens)
- **TTFT** (from 2 observations): avg ~2.28s (1 between 1-2.5s, 1 between 2.5-5s)
- **Time per output token**: 4 samples < 0.15s, 18 samples 0.15-0.2s, 2 samples 5-7.5s (likely trace compilation)
  - Steady-state TPOT: ~0.15-0.2s = **150-200ms ITL** (5-6.7 tok/s bs=1)
- **E2E latency** (1 completed request, 20 tokens): 12.07s
  - Prefill: 3.19s, Decode: 8.88s
  - Queue time: 2.2ms (negligible)
- **Prompt tokens processed**: 27 total
- **Generation tokens processed**: 26 total

---

## 8. Environment Config

### .env.glm47.galaxy File vs Container Env
**Match status**: CONSISTENT. All env file settings are correctly reflected in container environment.

### Key Configuration
| Setting | Value | Notes |
|---------|-------|-------|
| MESH_DEVICE | TG | Galaxy topology |
| VLLM_USE_V1 | 1 | V1 engine |
| trace_mode | decode_only | Traced decode |
| decode_trace_batch_buckets | [1,4,8,16,32] | 5 batch buckets |
| MAX_NUM_SEQS | 32 | Max batch |
| MAX_MODEL_LEN | 32768 | 32K context |
| GLM4_MOE_LITE_TP | 1 | TP enabled |
| GLM4_MOE_LITE_ENABLE_MOE | 1 | MoE enabled |
| GLM4_MOE_LITE_EP_L1 | 1 | L1 expert memory |
| GLM4_MOE_LITE_FUSE_EXPERTS_GATE_UP | 1 | Fused gate+up |
| GLM4_MOE_LITE_FUSE_QKV_A | 1 | Fused QKV |
| GLM4_MOE_LITE_CONCAT_HEADS | 1 | Concat heads |
| GLM4_MOE_LITE_MLA_USE_V_CACHE_SLICE | 1 | V cache slice |
| GLM4_MOE_LITE_DECODE_L1_ACT | 1 | L1 activations |
| GLM4_MOE_LITE_DENSE_TT_DTYPE | bf8 | BF8 weights |
| GLM4_MOE_LITE_EXPERTS_TT_DTYPE | bf8 | BF8 experts |
| GLM4_MOE_LITE_MTP | 0 | MTP disabled |
| GLM4_MOE_LITE_FUSED_KV_BRANCH | 0 | Megafusion off |
| GLM4_MOE_LITE_FUSED_PRE_SDPA | 0 | Megafusion off |
| TT_METAL_DEVICE_PROFILER | 0 | Tracy off |

---

## Summary of Findings

### NORMAL / HEALTHY
- All 32 Wormhole chips healthy: temps 36-40C, no faults, all DRAM OK, AICLK 1000 MHz
- No D-state processes
- No PCI/fabric errors in dmesg
- No stale UMD locks
- Container healthy and serving requests
- vLLM health endpoint returning 200
- Memory abundant (417 GiB available)
- Disk abundant (7 TB data, 868 GB root)
- IO minimal
- CPU load very light (3.5 / 64 cores)
- Environment config consistent between .env file and container

### CONCERNS
1. **104 restart count** on container. This matches the prior crash history (Task #1). Container has been stable for ~3 hours since last restart.
2. **Two orphaned tt-smi processes** consuming 99% CPU each (PIDs 82372, 588319). Not harmful to inference but wasting 2 CPU cores. Could be killed to free resources.
3. **Stale /dev/shm segments** from 8+ previous container incarnations (~350 MB). Harmless but could be cleaned up.
4. **Transient device init failure** at 21:39:14 (`board_id 0xffffffffffffffff`). Container recovered automatically. This is a known race condition during restart cycles.
5. **Early TPOT data suggests ~150-200ms ITL for bs=1** (~5-6.7 tok/s). Benchmark in progress will give definitive numbers.

### ACTIVE WORK
- `bench_decode.py` running: bs=1, gen=500, prefill_contexts=0
- 1 request currently in flight
