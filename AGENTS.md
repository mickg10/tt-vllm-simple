# GLM-4.7-Flash Performance Optimization: Team Plan

## Targets

- **Batch=1 decode:** 30 tok/s
- **Batch=32 decode:** 140+ tok/s aggregate
- **Benchmark matrix:** (1k/500, 10k/1000, 29k/3000 context/gen) × (batch=1,4,8,32)
- **Current baseline:** ~4.5 tok/s per user, ~28.5 aggregate at bs=8

## Team Roles

### Team Lead / Manager
- Coordinates work, assigns tasks, monitors progress
- Does NOT run tests or benchmarks (to preserve context)
- Reviews results from teammates, decides next step
- Uses Codex (gpt-5.2 model) for architectural decisions via `mcp__codex-cli__codex`

### Architect (MAINTAINS CONTEXT across steps)
- **Reads ALL MD files first** — perf-opt.md, PLAN_GLM47_FLASH.md, AGENTS.md, resume.md
- **Reviews what worked AND what did NOT work** from previous iterations
- Suggests the next optimization based on history
- Consults Codex (gpt-5.2 model, NOT default codex model) for its own suggestion
- **Compares own suggestion vs Codex suggestion**, picks the best
- **Updates the long-term plan** in perf-opt.md and PLAN_GLM47_FLASH.md
- **ONLY THEN** hands off to implementer with:
  - Specific implementation plan
  - What did NOT work previously (so implementer avoids dead ends)
  - Feature flag name and safe default
  - Expected improvement estimate
- The architect is the ONLY agent that keeps context across steps

### Implementer
- Makes code changes in tt-metal model files (moe_tt.py, decoder_layer_tt.py, etc.)
- Feature-flags all changes behind env vars with safe defaults
- Commits working changes

### Tester (Coherency Verification)
- **ALWAYS tests at tiny sizes FIRST before any benchmarking**
- Verification process:
  1. Run `python3 tests/run_coherency.py` — 32 tests in parallel
  2. Tests include: knowledge, math, logic, language, and 12 Python coding tasks
  3. Coding tests are validated by extracting the function and running it
  4. Test definitions: `tests/coherency_tests.json` — **MUST read this file**
  5. All 32 must PASS before proceeding to benchmarks
  6. For coding tasks, manually inspect generated code for reasonableness
- Reports pass/fail counts to team lead

### Benchmarker
- Runs the full benchmark matrix: `tests/bench_matrix.py`
- Records results in perf-opt.md with timestamps
- Compares before/after for each optimization step
- Reports aggregate numbers to team lead

### Debugger (as needed)
- Investigates failures, hangs, or crashes
- Reads container logs, profiles code, bisects issues
- Reports root cause and suggested fix

## Team Reuse Protocol

- The **same team** is reused step-to-step (no need to create a new team each iteration)
- Between steps, **clear context** for all agents EXCEPT the architect
- The **architect maintains context** across steps (keeps the full optimization history)
- Other agents (implementer, tester, benchmarker) get fresh context each step with a clear task description that includes what NOT to do (from architect's handoff)

## Testing Process (MANDATORY)

Every optimization step MUST follow this process:

1. **Implement** - Feature-flagged code change
2. **Container restart** - `docker compose --env-file dev/.env.glm47 -f dev/docker-compose.yml up -d --force-recreate vllm-tt`
3. **Wait for healthy** - Container must report healthy before any tests
4. **Coherency check** (Tester role):
   - `curl -s http://localhost:8088/v1/chat/completions -d '{"model":"zai-org/GLM-4.7-Flash","messages":[{"role":"user","content":"Hi"}],"max_tokens":32,"temperature":0}'`
   - Verify non-empty, coherent response text
   - `curl -s http://localhost:8088/v1/chat/completions -d '{"model":"zai-org/GLM-4.7-Flash","messages":[{"role":"user","content":"What is 7*8?"}],"max_tokens":32,"temperature":0}'`
   - Verify response contains "56"
5. **Benchmark** (Benchmarker role):
   - `python3 tests/bench_matrix.py`
   - Record results in perf-opt.md
6. **Commit** if improvement confirmed, revert if regression

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

## Key Files

- **Model code:** `tt-metal/models/demos/glm4_moe_lite/tt/`
- **Env config:** `docker_tt/dev/.env.glm47`
- **Perf log:** `/home/ttuser/src_docker/plan/glm47_flash/perf-opt.md`
- **Benchmark:** `docker_tt/tests/bench_matrix.py`
- **DeepSeek ref:** `tt-metal/models/demos/deepseek_v3/tt/`

## To Launch

```python
TeamCreate(team_name="glm-perf-sprint", description="GLM-4.7-Flash decode perf optimization")
```

Then spawn teammates per roles above and create tasks from the optimization backlog.
