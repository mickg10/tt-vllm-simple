# GLM-4.7-Flash Performance Optimization: Team Plan

The full team sprint plan lives outside git to survive resets:

**Canonical location:** `/home/ttuser/src_docker/plan/glm47_flash/claude_team_glm47_plan.md`

## Quick Reference

- **Target:** 30 tok/s decode (currently ~6 tok/s)
- **Team:** 8 agents via Claude Code `TeamCreate` (team-lead, architect, verifier, benchmarker, critic, tt-metal-impl, infra-impl, vllm-impl)
- **Workflow:** Architect proposes RFC (with Codex) -> Implement -> Review -> Verify (Qwen gate) -> Benchmark -> Decide
- **Backlog:** 6 RFCs prioritized by expected impact (expert parallelism first)

## To Launch

Read the full plan, then:

```
TeamCreate(team_name="glm-perf-sprint")
```

See `plan/glm47_flash/claude_team_glm47_plan.md` for complete role descriptions,
RFC backlog, launch script, and workflow diagram.
