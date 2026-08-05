# Longevity Clinical AI — evaluation report

Generated: `2026-08-05T12:52:45+00:00`

| Setting | Value |
|---|---|
| mcp_url | `http://127.0.0.1:9100/mcp` |
| cases | `1` |
| filter | `extra-compare-two-patients` |
| tier_b_model | `anthropic/claude-haiku-4.5` |
| judge_model | `openai/gpt-4o-mini` |
| repeats | `1` |
| temperature | `0.0` |

## Tier B

_agent in the loop via OpenRouter_

**Pass rate: 0.0%** (0 passed, 1 failed, 0 skipped, 0 errored of 1 runs)

| Category | Pass | Fail | Skip | Error | Pass rate |
|---|---:|---:|---:|---:|---:|
| multi_step | 0 | 1 | 0 | 0 | 0.0% |

| Axis | Pass | Fail | Skip | Pass rate |
|---|---:|---:|---:|---:|
| band_faithfulness | 2 | 0 | 0 | 100.0% |
| comparison | 1 | 0 | 0 | 100.0% |
| numeric_faithfulness | 2 | 1 | 0 | 66.7% |
| tool_selection | 2 | 0 | 0 | 100.0% |

### Failures

**`extra-compare-two-patients`** (multi_step)
- `no_fabricated_numbers` [numeric_faithfulness] expected `every patient value traceable to a tool result`, got `['5.7']` - untraceable: 5.7
