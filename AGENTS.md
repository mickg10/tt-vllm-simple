# GLM-4.7-Flash Performance Optimization: Team Plan

## Targets

- **Batch=1 decode:** 30 tok/s
- **Batch=32 decode:** 140+ tok/s aggregate
- **Benchmark matrix:** (1k/500, 10k/1000, 29k/3000 context/gen) x (batch=1,4,8,32)
- **Current baseline:** ~4.5 tok/s bs=1, ~27.8 tok/s bs=32 aggregate

## Optimization Loop Workflow

```
ARCHITECT (long-lived, keeps context across all steps)
  |-- Reads perf-opt.md history + what worked/failed
  |-- Consults Codex (gpt-5.2 via mcp__codex-cli__codex)
  |-- Decides next optimization step
  +-- Hands off to implementer with clear spec + what NOT to do

IMPLEMENTER (ephemeral, ONE at a time, never concurrent)
  |-- Implements the change
  |-- Runs smoke tests (coherency)
  |-- Reports back, shuts down
  +-- If smoke fails -> debug, fix, retry

BENCHMARKER (ephemeral, after smoke passes, never concurrent with implementer)
  |-- Runs full benchmark matrix
  |-- Reports numbers, shuts down
  +-- Results go back to architect

ARCHITECT evaluates results
  |-- Improvement? -> Commit, record in perf-opt.md, design next step
  +-- No improvement? -> Revert, record failure, try different approach
```

## CRITICAL RULES

1. **ONE implementer at a time** -- NEVER run concurrent implementers. They edit the same files and restart the same container, causing corruption.
2. **Architect is the only long-lived agent** -- maintains context across all optimization steps
3. **Implementers are ephemeral** -- spawned per task, shut down when done
4. **Benchmarker runs AFTER implementer finishes** -- never concurrent
5. **Batch-adaptive strategy** -- different optimizations may work for different batch sizes. Test BOTH bs=1 and bs=32.
6. **Feature-flag everything** -- new optimizations behind env vars with safe defaults
7. **Record everything in perf-opt.md** -- what worked, what didn't, exact numbers

## Key Files

- **Model code:** `tt-metal/models/demos/glm4_moe_lite/tt/`
- **Env config:** `docker_tt/dev/.env.glm47`
- **Perf log:** `/home/ttuser/src_docker/plan/glm47_flash/perf-opt.md`
- **Benchmark:** `docker_tt/tests/bench_matrix.py`
- **Coherency:** `docker_tt/tests/run_coherency.py` (32 tests)
- **DeepSeek ref:** `tt-metal/models/demos/deepseek_v3/tt/`

## Codex Usage

**Always use the MCP tool** `mcp__codex-cli__codex` (NOT the Codex CLI command).
**Always use gpt-5.2 model** (NOT the default codex model).
```python
mcp__codex-cli__codex(
    prompt="...",
    model="gpt-5.2",
    cwd="/home/ttuser/src_docker/ws/glm47_flash"
)
```

**IMPORTANT: Codex queries can easily take 20+ minutes on complex tasks.** This is normal — do NOT assume the agent is stuck just because Codex hasn't returned yet. Be patient and wait for the response.

## Testing Process

1. **Implementer runs smoke tests:** `python3 tests/run_coherency.py` (32 tests, all must match baseline 30/32)
2. **Benchmarker runs:** `python3 tests/bench_matrix.py --only-ctx 1000` (bs=1 and bs=32 at minimum)
3. Results recorded in perf-opt.md with timestamps

## To Launch

```python
TeamCreate(team_name="glm-perf-sprint", description="GLM-4.7-Flash decode perf optimization")
```

Then spawn architect (long-lived), and implementer/benchmarker (ephemeral, one at a time) per workflow above.

## Benchmark History

See **Section 6.4** of `PLAN_GLM47_FLASH.md` for the full benchmark history table (updated after every approach).
