# Longevity Clinical AI — evaluation report

Generated: `2026-08-04T16:59:37+00:00`

| Setting | Value |
|---|---|
| mcp_url | `http://127.0.0.1:9100/mcp` |
| cases | `21` |
| filter | `(none)` |

## Tier A

_deterministic, MCP tools called directly, no LLM_

**Pass rate: 100.0%** (18 passed, 0 failed, 3 skipped, 0 errored of 21 runs)

| Category | Pass | Fail | Skip | Error | Pass rate |
|---|---:|---:|---:|---:|---:|
| citation | 1 | 0 | 0 | 0 | 100.0% |
| multi_step | 2 | 0 | 0 | 0 | 100.0% |
| numeric_faithfulness | 6 | 0 | 0 | 0 | 100.0% |
| safety | 2 | 0 | 3 | 0 | 100.0% |
| tool_selection | 6 | 0 | 0 | 0 | 100.0% |
| trend | 1 | 0 | 0 | 0 | 100.0% |

| Axis | Pass | Fail | Skip | Pass rate |
|---|---:|---:|---:|---:|
| band_faithfulness | 14 | 0 | 0 | 100.0% |
| citation | 0 | 0 | 1 | n/a |
| comparison | 1 | 0 | 0 | 100.0% |
| determinism | 1 | 0 | 0 | 100.0% |
| numeric_faithfulness | 17 | 0 | 0 | 100.0% |
| safety | 0 | 0 | 5 | n/a |
| tool_contract | 20 | 0 | 0 | 100.0% |
| trend | 1 | 0 | 0 | 100.0% |

No failures.
