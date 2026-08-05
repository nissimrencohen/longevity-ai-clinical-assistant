# SOLUTION

A clinical chat assistant for a longevity clinic: LibreChat → MCP → FastAPI →
MLflow + database, with five disease risks computed live from a patient's
biomarkers, explained, and evaluated.

**Repository:** https://github.com/nissimrencohen/longevity-ai-clinical-assistant
**LibreChat release pinned:** `v0.8.7` (config schema `1.3.13`)

At a glance:

| | |
|---|---|
| Tests | **306 passed, 0 skipped, 0 failed** |
| Tier A evals (deterministic, free) | **100%** — 25 pass, 0 fail, 2 behavioural-skip of 27 |
| Tier B evals (agent in the loop) | **100%** on the last *full* run — 21 cases × 3 repeats, 0 errored ([report](evals/results/tier-b-baseline.md)) |
| Lint | `ruff` clean |
| Boot | `docker compose up -d --wait` → six healthy services (plus a one-shot seed) in ~25s |

---

## 1. Core requirements and bonuses

The six tasks from `README.md`, and where each lives:

| # | Task | Done | Where |
|---|---|---|---|
| 1 | Backend — two endpoints, per-model feature vectors | ✅ | [`backend/app/api/v1/endpoints.py`](backend/app/api/v1/endpoints.py), [`services/features.py`](backend/app/services/features.py), [`services/risk.py`](backend/app/services/risk.py) |
| 2 | MLflow — five models served on `:5001` | ✅ | [`models/register_router.py`](models/register_router.py) — one pyfunc routing on `params["model"]` |
| 3 | MCP — bearer-authed tools | ✅ | [`mcp-server/server.py`](mcp-server/server.py) — five tools, not two |
| 4 | LibreChat — running and wired | ✅ | [`librechat/librechat.yaml`](librechat/librechat.yaml), [`librechat/SETUP.md`](librechat/SETUP.md) |
| 5 | Evals — tool-call correctness, numeric faithfulness | ✅ | [`evals/`](evals/) — two tiers, 27 cases, [`HARNESS.md`](evals/HARNESS.md) |
| 6 | **Bonus** — retrieval with citations | ✅ | [`services/guidelines.py`](backend/app/services/guidelines.py) — TF-IDF *and* MiniLM embeddings, every citation verified on disk |
| 6 | **Bonus** — custom agent | ✅ | [`agent/router.py`](agent/router.py) — deterministic tool-chaining orchestrator |

Both bonus tracks are done, not one. Everything below is why the choices were made.

### Async Python, end to end

Every path from HTTP entry to database and model server is `async`. There is no
synchronous escape hatch and no thread-pool bridge.

- `aiosqlite` for SQLite; **SQLAlchemy 2.0 async + asyncpg** for Postgres. Never
  stdlib `sqlite3` in a request path.
- One pooled `httpx.AsyncClient`, owned by the app lifespan
  ([`backend/app/main.py`](backend/app/main.py)), not created per request —
  per-request clients throw away connection pooling, which is the usual reason
  "async" code turns out slow.
- The SQLite connection is **not held across the model calls**. The endpoint
  reads, closes, does its network I/O, then reopens to write. Holding a
  write-capable connection across a 200 ms model round trip is how a slow
  upstream becomes a locked database.

### Concurrent model calls

The five risks are independent, so they are fired together:

```python
scored = await asyncio.gather(
    *(self._score(spec, payloads[spec.risk_code], hashes[spec.risk_code])
      for spec in MODEL_SPECS)
)
```

[`backend/app/services/risk.py`](backend/app/services/risk.py). Under
observability the five appear as sibling spans, so the claim is *provable* rather
than asserted.

### Typed responses, and clean 404 / 502

Pydantic models throughout ([`backend/app/schemas.py`](backend/app/schemas.py)).
The error contract is deliberately granular, because each code means something
different to the assistant:

| Code | Meaning |
|---|---|
| **403** | Known caller, not permitted. Distinct from 404: answering "no such patient" to an unauthorised caller is a small lie, and telling them "exists but forbidden" leaks existence. The denial is audited. |
| **404** | Unknown patient — so the assistant can say so instead of inventing one. |
| **422** | Patient exists, a required model input is missing. We refuse to score rather than impute a lab value. |
| **502** | Model server unreachable, or returned something that is not a probability. |

### LLM app building — tool contracts

Five MCP tools over streamable HTTP with static bearer auth. Tool docstrings are
written **for the model, not for humans** — they state when to call, what an
argument looks like, and what the caller must do with the result. Backend
failures become structured, actionable tool errors that explicitly instruct
against fabricating values:

> "No patient with ID P999 exists in the clinic database. Do not report any
> biomarkers, risks or trends for this patient, and do not substitute a different
> patient — say that the record was not found."

Two wiring faults worth recording, because both present as *"MCP connected,
0 tools"* and cost hours:

1. **The trailing slash is inverted under FastMCP 3.x.** `GUIDE.md` says the URL
   needs one — true for 2.x. Under 3.4.4 the canonical path is `/mcp` and `/mcp/`
   answers `307`, and LibreChat does not replay the `Authorization` header across
   that redirect.
2. **`requiresOAuth: false` is mandatory.** LibreChat probes the URL with *no*
   headers; FastMCP answers `401` with a `WWW-Authenticate` challenge; LibreChat
   concludes the server speaks OAuth and never sends the static token again. I
   isolated this by hardcoding the token — it *still* 401'd — which ruled out
   `${MCP_BEARER_TOKEN}` interpolation and pointed at the probe.

### Evaluation mindset

Two tiers, split by what they need and what they can prove
([`evals/HARNESS.md`](evals/HARNESS.md)).

**Tier A** — deterministic, no LLM, no API key, ~2s. Calls the MCP tools directly
and asserts the values the assistant would be repeating: exact biomarkers,
probability tolerances, bands, trends, horizons, drivers, citations, the
unknown-patient contract, and determinism. Because it is free it can gate every
commit, which is what makes it a gate at all.

**Tier B** — an OpenAI-compatible tool-calling loop against OpenRouter with the
real MCP tools attached. Scores tool selection, numeric faithfulness, band
faithfulness, trend, explanation faithfulness, and safety.

The numeric-faithfulness rule is explicit: **a number in the prose must trace to
a tool result, the doctor's question, or the assistant's own instructions.**
Anything else fails. Only behavioural `safety` facts go to an LLM judge — a judge
has no business grading arithmetic.

Statuses distinguish `skip` (not assertable at this tier) from `error`
(infrastructure). An errored run is excluded from the pass rate: the first Tier B
run counted HTTP 402s as model failures and reported 44%; the same run reads
95.2% once infrastructure is separated out.

Fourteen cases were added beyond the gold set, for failure modes it does not
reach — hallucination *mid-conversation* after a successful lookup, nearest-name
substitution, out-of-scope refusal, two-patient comparison, both
NULL-`gestational_diabetes` patients, the null T2DM horizon, and determinism.

### Correctness over guessing

The system refuses rather than guesses, in several specific places:

- A **missing lab raises** instead of defaulting to zero — imputing an eGFR would
  invert CKD risk.
- The **one sanctioned imputation is documented and audited**:
  `gestational_diabetes` is NULL for all four male patients, and sklearn raises
  on NaN, so it is coalesced to `0` — "not applicable", not "unknown" — with the
  substitution recorded in `inputs_json`.
- A **model-server outage fails the whole panel (502)** rather than returning
  four of five risks: a missing risk reads as "not elevated".
- Any value outside `[0, 1]` is rejected as not-a-probability, guarding the
  `predict` vs `predict_proba` trap.
- `find_patient` returns **no match** for "Miriam Cohen" rather than falling back
  on Maya Cohen, the only Cohen in the clinic. Near-misses are the dangerous case:
  the wrong patient's labs, delivered confidently.

### Bonus — retrieval with citations

`search_guidelines` over `data/guidelines/`, chunked at **heading level** because
the `##` sections are the unit a clinician would cite.

**The citation, not the retriever, is the point.** A plausible citation to text
that does not exist is worse than none — it launders an invented claim as a
sourced one. Every chunk carries its source file, heading and exact line span,
and `verify_citation` re-reads the file on disk. That is deterministic and free,
so **Tier A verifies every retrieved snippet** rather than asking a model whether
a citation looks right. Tests cover the three failure modes separately: invented
file, real file with invented heading, and real heading with a paraphrase beyond
the source — the last being likeliest and hardest to spot.

Both backends work and are tested: **TF-IDF (default)** and **MiniLM embeddings
via Chroma** (`uv sync --extra rag`).

### Bonus — custom agent

[`agent/router.py`](agent/router.py) drives the same MCP server: resolve →
risks → ground the elevated ones in cited guidance. It uses **no LLM**, which is
the point rather than a shortcut — for a routine that should run identically
every time, a graph is the right shape. It cannot hallucinate (it never generates
prose), it runs free in CI, and an ambiguous name **stops the run** instead of
picking the first match.

---

## 2. Architectural additions, and why they matter clinically

### PostgreSQL — for the write, not for fashion

The risk endpoint **writes on every request**. SQLite takes a database-wide write
lock, so two doctors asking about two *different* patients serialise. More
importantly the dedupe rule ("append only when the inputs changed") is a
read-then-write race under SQLite; in Postgres it is a partial unique index over
`(patient_id, model_name, inputs_hash)`, so the append is a single atomic
`INSERT … ON CONFLICT DO NOTHING`. `append_risks` returns only the rows actually
inserted, so a losing racer reports `persisted=false` truthfully.

*Found by running it:* Postgres will only infer a **partial** unique index when
the statement repeats the index predicate. Without `index_where` it reports "no
unique or exclusion constraint matching the ON CONFLICT specification" — every
risk request 500'd until Tier A caught it within seconds.

### Redis — keyed on the payload hash, not the patient

The models are deterministic and pure, so identical inputs imply an identical
probability: **a cache hit cannot serve a wrong answer.** The hash covers model
name and version, so re-registering a model invalidates its cache automatically.
It is the same hash the dedupe uses — one primitive, two requirements.

What a cache *can* get wrong is provenance, so a hit is labelled: `source:
"cache"` and `computed_at` is the **original** computation time. An hour-old
number presented as fresh is a safety problem, not a performance one. Redis fails
open — an outage degrades latency, never correctness.

### The output guardrail proxy — enforcement, not instruction

Asked whether to start atorvastatin, the assistant issued a definitive
recommendation in **4 of 7 observed runs**, and strengthening the prohibition in
all three places it was stated made the rate *worse*. Prompts are advisory.

The obvious homes for a guard — the backend, the MCP server — are both wrong:
neither ever sees the assistant's prose. LibreChat's configurable `baseURL` gives
a real interception point:

```
LibreChat ──▶ guard :8080 ──▶ OpenRouter
```

Result: **2 of 3 failed without it; 3 of 3 passed through it.** Streaming is
buffered deliberately — you cannot retract tokens already on screen, so
mid-stream inspection enforces nothing.

Its first live intervention taught the design something. It removed:

> *"Whether to initiate atorvastatin 40 mg daily—or another intensity—depends on
> your assessment of his overall risk, his preferences, any contraindications…"*

That defers correctly; its only fault is naming an agent and dose. Deleting the
sentence threw away the useful part, so hedged sentences are now **redacted, not
deleted** — the prescription becomes "this medication", the reasoning survives.

### SHAP with the maths proved, not assumed

For a linear model the Shapley value is closed-form: `φⱼ = wⱼ(xⱼ − x_refⱼ)`. One
vector subtraction — no sampling, no latency cliff, which is why every risk on
every request can afford an explanation.

Two proofs, not assertions
([`backend/tests/test_explanations.py`](backend/tests/test_explanations.py)):

- **Additivity** — `base + Σφ == logit(p)` to 1e-9, across 5 models × 8 patients.
- **Shapley equivalence** — the closed form is checked against a **brute-force
  computation enumerating every coalition** and averaging marginal contributions
  over all orderings, plus an efficiency check on the brute force itself so the
  oracle is trustworthy rather than two implementations of one mistake.

The reference is each model's own calibration anchor — a healthy 35-year-old —
imported from `generate_models.py` and persisted **inside the model artefact**. A
data-derived background would make explanations depend on who else is in the
database.

**The failure mode explanations introduce:** contributions are additive in
**log-odds, not probability**. "Elevated BMI adds 12% to her risk" is false
however natural it sounds. Three layers forbid it, and a scorer fails any answer
that does it — while deliberately *not* flagging a legitimate probability quoted
as a percentage, which was the harder half.

### Two-way PHI scrubbing

Every earlier phase controlled what the *tools* return but not what the doctor
types. The guard proxy sees the whole request body, so it can rewrite both ways:

```
doctor  "What is Maya Cohen's eGFR?"
  │ scrub
model   "What is Patient Zxsyqn's eGFR?"    ← all OpenRouter ever sees
  │ restore
tools   find_patient("Maya Cohen")          ← MCP gets the real name
```

Tool **results** are scrubbed too, not just user turns — MCP returns the real name
and LibreChat feeds it straight back, so a user-message-only scrubber leaks on the
very next turn.

*This is where the final audit earned its keep.* The first token format,
`[PATIENT_7F3A2C]`, **broke tool calling in 3 of 3 live cases**: the model read it
as a patient ID, skipped `find_patient`, and invented `patient_id="P084"` from the
token's own hex digits. Tokens are now name-shaped with **no digits**
("Patient Zxsyqn"). The next run showed the model reading "Patient" as a title and
passing only "Zxsyqn", so restore now matches the bare core too. Local tests could
not have caught either — they proved the round trip, not how a model *reads* a
token.

### RBAC and audit

The brief says "all doctors can see all patients". That is a legitimate model for
one small clinic — the flaw was that it was **implicit**. The default
(`RBAC_MODE=clinic_wide`, role `physician`) behaves exactly as specified; what is
new is that the policy is explicit, configurable and audited.

`READ_RISKS` and `PERSIST_RISKS` are separate actions: a nurse sees the full panel
but does not write to the trend, because the trend is a clinical record. Identity
comes from the **verified** MCP token, so a caller cannot assert a role. Every
decision is audited — **denials included**, with reasons.

### Deterministic retrieval over blind embeddings

At five short documents, TF-IDF is competitive with embeddings and has the large
advantage of being **reproducible**, so it can run in CI and in the free eval
tier. Embeddings are implemented, tested and one config value away
(`RETRIEVAL_BACKEND=embedding`); they earn their keep when the corpus grows. The
protocol makes that a config change, not a rewrite.

### Observability that cannot leak

Auto-instrumentation records URLs, query strings and exception messages as span
attributes — so `?patient_id=P004` and lab values would land in a third-party
dashboard, undoing the PHI work while *looking* like an improvement. The exporter
is wrapped: allowlist (deny by default), query strings stripped, patient ids
pseudonymised, span events dropped. Spans are **rebuilt, not mutated** —
`ReadableSpan` is meant to be immutable, and editing private state fails silently
on upgrade with PHI leakage as the failure mode.

---

## 3. How to run it, and the feature flags

### Docker (recommended)

```bash
cp .env.example .env
docker compose up -d --wait
```

`--wait` blocks until every healthcheck passes, so a green exit means the stack is
actually serving rather than merely started. Six services — `postgres`, `redis`,
`mlflow`, `backend`, `mcp`, `guard` — plus a one-shot `seed` container that loads
the patient data into Postgres and exits.

```bash
make up        # same thing          make logs     # tail all services
make ps        # status + health     make down     # stop
make rebuild   # after code changes
```

Only two host ports are published — `9100` → `mcp:9000` for LibreChat, and `9200`
→ `guard:8080` for the safety proxy. Postgres, Redis, MLflow and the backend are
**not reachable from the host at all**; they talk over a private bridge network by
service name. Attach LibreChat to `longevity-net` (see
[`librechat/docker-compose.network.yml`](librechat/docker-compose.network.yml))
and even `9100` becomes unnecessary.

### LibreChat

Pinned to **`v0.8.7`**. Copy both files into the checkout root:

```bash
cp librechat/.env <librechat-checkout>/.env
cp librechat/librechat.yaml <librechat-checkout>/librechat.yaml
```

Set `OPENROUTER_KEY` in that `.env`, then:

```bash
docker compose -f deploy-compose.yml up -d
```

Open http://localhost:3080, register, create an Agent on the **OpenRouter**
endpoint with a tool-capable model, enable the `longevity-clinical` tools, and
paste [`librechat/AGENT_INSTRUCTIONS.md`](librechat/AGENT_INSTRUCTIONS.md).

To route the chat through the safety guard, point the endpoint `baseURL` at
`http://host.docker.internal:9200/v1`.

### Evals

```bash
uv run python evals/harness.py --tier a                 # deterministic, free
uv run python evals/harness.py --tier both --repeats 3  # needs OPENROUTER_KEY
```

### Feature flags — reverting to the baseline

**Every upgrade is additive and defaults to the assignment's behaviour.** A fresh
clone runs `pytest` and host mode with **no services at all**:

| Flag | Process default | Effect |
|---|---|---|
| `DB_BACKEND` | `sqlite` | `postgres` to use the SQLAlchemy/asyncpg store (compose sets this) |
| `CACHE_BACKEND` | `none` | `redis` to enable the risk cache (compose sets this) |
| `RBAC_MODE` | `clinic_wide` | `care_team` restricts to assigned patients |
| `AUDIT_ENABLED` | `true` | every access decision recorded |
| `RETRIEVAL_BACKEND` | `lexical` | `embedding` for Chroma/MiniLM |
| `OTEL_ENABLED` | `false` | `true` + `--profile observability` for Phoenix |
| `GUARD_PHI_DEIDENTIFY` | `true` | inbound name scrubbing |
| `GUARD_PHI_FAIL_CLOSED` | `false` | refuse traffic if the term list is unavailable |

**To run exactly the baseline the brief describes** — SQLite file, no cache, no
containers — use host mode, where `sqlite`/`none` are already the defaults:

```bash
powershell -ExecutionPolicy Bypass -File scripts/run_stack.ps1
```

The script starts MLflow, the backend and the MCP server as detached host
processes and waits for each to answer.

`CACHE_BACKEND=none` also works unchanged inside Docker. `DB_BACKEND=sqlite` does
**not**, deliberately: compose mounts `./data` **read-only**, because with
Postgres live the SQLite file is a seed fixture rather than a write target, and a
read-only mount is what stops a misconfigured backend from silently writing to the
wrong store. Setting `DB_BACKEND=sqlite` in a container therefore fails loudly
("attempt to write a readonly database") instead of appearing to work. Drop the
`:ro` suffix if you want that combination.

`/health` reports which pair is live, so a misconfigured deployment is visible
from the healthcheck rather than from surprising results.

### Test suite

```bash
uv run pytest                    # 299 passed, 7 conditionally skipped
make up-debug && POSTGRES_DSN="postgresql+asyncpg://clinic:clinic@127.0.0.1:55432/clinic" \
  uv run pytest                  # 306 passed, 0 skipped
```

The 7 skips are integration tests needing MLflow and Postgres on host ports; the
debug overlay publishes them.

---

## 4. Trade-offs and what is left

### The GET that writes

`GET /api/v1/patients/{id}/risks` computes five probabilities and **appends them
to the `risks` table**. That is a side effect on a method HTTP defines as safe, so
a crawler, a browser prefetch or a proxy retry can grow the clinical record.

I kept it, because the alternative is worse for this system's actual purpose. The
assistant asks the same question repeatedly during a conversation; making it a
`POST` would force the MCP tool to choose between "read" and "record" on every
turn, and it would get that choice wrong. What I did instead is make the write
**idempotent by content**: a row appends only when the input vector's hash is new,
enforced in Postgres by a partial unique index rather than by an application
check. Ten identical GETs produce one row, so the operation is *effectively* safe
even though the method's contract is being stretched.

The honest version is: this is a deliberate violation with a compensating control,
not an oversight. The right shape is `POST …/risk-computations` for the write and
a pure `GET` for the read, and that is first on the list below.

### Assumptions I had to make

| Assumption | Why, and what breaks if it is wrong |
|---|---|
| **"Today" is fixed at 2026-07-09** (`CLINIC_TODAY`) | Ages are derived against the date in `DATA_DICTIONARY.md`, not the wall clock. Using `date.today()` would make every gold probability drift out of tolerance as the year advances, and the eval suite would rot silently. Real deployments need a real clock — and a different way of pinning gold values. |
| **Units pass through unconverted** — cm, kg, mg/dL, mmHg, mL/min/1.73m² | The pickles were trained on the data dictionary's units, so converting would be the bug. There is no unit metadata in the models, so this is an assumption enforced by the round-trip test (P004 CKD = 0.5000), not something the artefact tells us. A lab feeding mmol/L cholesterol would score wrong and *look* fine. |
| **BMI and waist–hip ratio are derived, not stored** | `bmi = kg / m²` and `waist_cm / hip_cm`. Both are model inputs with no column behind them; if either component is missing the request 422s rather than imputing. |
| **`former` smokers are not `current_smoker`** | The flag is present-tense. Counting former smokers would inflate CVD and CLD risk for exactly the patients most likely to ask. |
| **Dipstick `trace`/`1+`/`2+`/`3+` all mean proteinuria = 1** | Only `negative` is 0. Treating `trace` as negative is the defensible alternative and would lower CKD risk for borderline patients — the choice is documented rather than silent. |
| **`gestational_diabetes` NULL → 0** | NULL for all four male patients, and sklearn raises on NaN. Coalesced to 0 as "not applicable", recorded in `inputs_json`. This is the only imputation in the system, and it is the one place where "not applicable" and "unknown" genuinely coincide. |

**OpenRouter is not HIPAA-eligible and will not sign a BAA.** No amount of
application-layer engineering changes that. The de-identification boundary means
a real patient *name* no longer leaves the trust boundary, which is a genuine
improvement, but it is not compliance. A real deployment moves the LLM tier to a
covered provider or an in-VPC model. That is procurement, not code.

**PHI scrubbing covers what it knows about.** Patient names from the clinic
roster. A doctor who types a date of birth, an address, or free-text detail is not
protected, and no regex will fix that. It closes the specific, predictable leak
this application creates; it is not a general PHI firewall.

**The prescribing guard is a heuristic.** Drug detection is a small lexicon plus
stem suffixes (`-statin`, `-pril`, `-sartan`), not a formulary — it will miss
unusual agents. The dose+frequency rule catches most of what the lexicon does
not. In production this wants a real drug vocabulary (RxNorm) behind the same
interface.

**Tier B numbers are model-dependent, lightly sampled, and not fully current.**
The 100% figure is 21 cases × 3 repeats of `claude-haiku-4.5`; the six cases added
afterwards (explanations, `find_patient`) have been verified individually at Tier B
but not in a full sweep — Tier A covers them deterministically and a full sweep
costs real money on every run. The suite has caught a genuine safety failure that a green run
did not reproduce (the prescribing case, ~1 in 3 before the guard) — **a green run
is weak evidence; a recorded failure is strong evidence.** Three repeats cannot
characterise a 1-in-3 failure.

**Scorer thresholds were tuned after seeing failures**, which is a mild form of
fitting to the test set. Every refinement is principled and pinned by a test that
verifies the scorer still catches genuine fabrications, but the ordering matters
and a reviewer should know it.

**The LLM judge is not validated against human labels.** Its verdicts and
reasoning are stored so agreement *could* be measured; it has not been. It also
got one call wrong that a deterministic check got right — it failed a correct
answer by deciding "contributes a log-odds of 1.04" was a percentage-point amount
of risk, which is why direction checks moved out of the judge.

**Not done, and I would do next, in order:** validate the judge against my own
labels; replace the drug lexicon with RxNorm; add `POST
/api/v1/patients/{id}/risk-computations` as the semantically correct write (the
GET is idempotent-by-inputs, but it is still a GET that writes); wire the audit
log to an append-only Postgres grant rather than relying on the application; and
characterise the prescribing failure rate properly with ~20 repeats.

### Where AI tooling was used, and what I rejected

Claude Code was used throughout — for scaffolding, for the eval harness, and as a
reviewer of my own designs. Three things I want to be concrete about, because
they are the interesting part:

**What it got wrong, and I caught by running the system.** The PHI token format
was its design and mine; local tests all passed and it broke tool calling in 3 of
3 live cases. The lesson generalises: unit tests proved the round trip, but not
how a *model reads* a token. Same story for the guard deleting a well-formed
deferral, and for the retriever's top hit being a disclaimer.

**What I rejected.** A LangGraph agent for the custom-agent track — it would have
added a dependency and an API cost to prove orchestration that a plain graph
proves better and for free. Chroma as the *default* retriever, because at five
documents it buys nothing over TF-IDF and costs determinism. Langfuse as the
default trace backend, because self-hosting it is four more containers.

**What the evals caught that review would not have.** The `ON CONFLICT` partial
index bug, the corrupted number `0.018510.0185`, the `"ris"` stem matching
`"risk"`, and the PHI token breaking tool calls. Every one of those looked correct
in the diff.
