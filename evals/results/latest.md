# Longevity Clinical AI — evaluation report

Generated: `2026-08-05T22:15:58+00:00`

| Setting | Value |
|---|---|
| mcp_url | `http://127.0.0.1:9100/mcp` |
| cases | `28` |
| filter | `(none)` |

## Tier A

_deterministic, MCP tools called directly, no LLM_

**Pass rate: 100.0%** (26 passed, 0 failed, 2 skipped, 0 errored of 28 runs)

| Category | Pass | Fail | Skip | Error | Pass rate |
|---|---:|---:|---:|---:|---:|
| citation | 3 | 0 | 0 | 0 | 100.0% |
| explanation | 3 | 0 | 0 | 0 | 100.0% |
| multi_step | 2 | 0 | 0 | 0 | 100.0% |
| numeric_faithfulness | 7 | 0 | 0 | 0 | 100.0% |
| safety | 3 | 0 | 2 | 0 | 100.0% |
| tool_selection | 7 | 0 | 0 | 0 | 100.0% |
| trend | 1 | 0 | 0 | 0 | 100.0% |

| Axis | Pass | Fail | Skip | Pass rate |
|---|---:|---:|---:|---:|
| band_faithfulness | 16 | 0 | 0 | 100.0% |
| citation | 4 | 0 | 1 | 100.0% |
| comparison | 1 | 0 | 0 | 100.0% |
| determinism | 1 | 0 | 0 | 100.0% |
| explanation | 13 | 0 | 3 | 100.0% |
| numeric_faithfulness | 20 | 0 | 0 | 100.0% |
| phi | 1 | 0 | 0 | 100.0% |
| safety | 0 | 0 | 5 | n/a |
| tool_contract | 29 | 0 | 0 | 100.0% |
| trend | 1 | 0 | 0 | 100.0% |
| unknown | 0 | 0 | 1 | n/a |

No failures.
