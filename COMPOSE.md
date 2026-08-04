# Running the stack

Two supported modes. Docker is the default; host mode is kept for fast iteration
and debugging.

## Verified

| Check | Result |
|---|---|
| `docker compose up -d --wait` from a torn-down state (network removed) | all three healthy in **23s** |
| Tier A evals against the containerised stack | **100%** (18P / 0F / 3 skipped) |
| Tier B evals against the containerised stack, 3 repeats | **100%** (63P / 0F / 0 errored) |
| Self-healing after the app process dies | back to `healthy` in ~12s, `RestartCount` 0 -> 1 |
| LibreChat tool discovery | `Tools: ping, get_current_biomarkers, get_current_risks` |
| Host mode still works | Tier A **100%** |
| `pytest` against the containers (with `make up-debug`) | **104 passed, 0 skipped** |

Same eval numbers as the pre-containerisation baseline, which is the point: the
refactor moved the runtime without moving the behaviour.

## Docker (recommended)

```bash
docker compose up -d --wait
```

That is the whole thing from a clean clone. `--wait` blocks until every
healthcheck passes, so a green exit means the stack is actually serving, not just
that containers were created.

```bash
make up        # same thing
make ps        # container status + health
make logs      # tail all three
make down      # stop
make rebuild   # after changing code or regenerating models
```

### What it starts

| Service | Internal address | Published |
|---|---|---|
| `mlflow` | `mlflow:5001` | no |
| `backend` | `backend:8001` | no |
| `mcp` | `mcp:9000` | `${MCP_PUBLISH_PORT:-9100}` -> 9000 |

All three carry `restart: always` and a healthcheck, and start in dependency
order (`mlflow` healthy -> `backend` healthy -> `mcp`).

### Why only one published port

The services address each other by **service name** on a private bridge network
(`longevity-net`), so none of their traffic touches a host port. This is not
tidiness — both failure modes it prevents actually happened while building this:

* Three host processes were killed by the shell that launched them, silently,
  mid-session. Containers with `restart: always` do not have a launching shell.
* An unrelated QuestDB container from another project published port 9000 and
  took it while our MCP server was down. LibreChat then connected to QuestDB and
  reported "Failed to initialize MCP server". Container-to-container traffic on a
  private network cannot be intercepted this way.

The one published port exists for two callers that live outside the network: the
eval harness (a host-run tool) and the LibreChat container. Move it with
`MCP_PUBLISH_PORT=9200 docker compose up -d` if it ever collides.

### Running the full test suite against the containers

Isolation has one cost worth naming: `backend/tests/test_mlflow_integration.py`
talks to `http://127.0.0.1:5001/invocations`, so under the default stack those
four tests **auto-skip** — and you quietly lose the coverage that proves the
model server returns probabilities rather than class labels.

The debug overlay re-publishes backend and MLflow so they run:

```bash
make up-debug          # or: docker compose -f docker-compose.yml -f docker-compose.debug.yml up -d --wait
uv run pytest          # 104 passed, 0 skipped
```

Don't leave it on — that puts you back to competing for host ports with
everything else on the machine, which is the problem the default stack solves.

### Removing the last host port

Attach LibreChat to the same network and even that port becomes unnecessary:

```bash
docker compose up -d                                  # in this repo, first
cd <librechat-checkout>
docker compose -f deploy-compose.yml \
               -f <this-repo>/librechat/docker-compose.network.yml up -d
```

Then set `url: http://mcp:9000/mcp` in `librechat.yaml` (the line is already
there, commented) and drop the `mcpSettings.allowedAddresses` entry — a
container hostname on a shared network is not a private-IP SSRF target.

### Notes on the image

One image serves all three roles; they differ only in the command. Three images
would mean three copies of scikit-learn, pandas and mlflow for no benefit.

The MLflow router artefact is **built into the image**. `models/mlflow_risk_router`
is generated from the committed pickles and is gitignored, so a clean clone has
nothing to serve — the build runs `models/register_router.py` and round-trips the
P004 CKD anchor (expected exactly 0.50) as it goes, so a bad build fails loudly.
That is also why `models/` is deliberately not bind-mounted: the host directory
would shadow the artefact.

`data/` **is** bind-mounted. The risks table is an append log the backend writes
to on every request, and keeping it on the host means it survives rebuilds and
stays visible to `pytest` and `make db`.

## Host mode

Still fully supported — it is faster to iterate against and easier to attach a
debugger to.

```bash
powershell -ExecutionPolicy Bypass -File scripts/run_stack.ps1          # start
powershell -ExecutionPolicy Bypass -File scripts/run_stack.ps1 -Status
powershell -ExecutionPolicy Bypass -File scripts/run_stack.ps1 -Stop
```

Or one service at a time: `make mlflow`, `make backend`, `make mcp`.

Host mode reads ports and URLs from the repo `.env` (`BACKEND_URL`,
`MLFLOW_URL`, `MCP_PORT`), which point at `127.0.0.1`. Compose overrides those
with service names, so the two modes do not interfere — but **do not run both at
once**: they compete for the same published MCP port.

## Evals against either mode

The harness talks to the MCP server over HTTP and does not care which mode is
running:

```bash
uv run python evals/harness.py --tier a        # deterministic, no API key
uv run python evals/harness.py --tier both --repeats 3
```

It reads `MCP_PORT` from `.env`; override with `MCP_EVAL_URL` if you publish the
MCP server somewhere else.

## Troubleshooting

**`docker compose up` fails to bind the MCP port.** Something else has it. Find
it with `netstat -ano | findstr :9100`, then either stop that process or set
`MCP_PUBLISH_PORT`.

**Healthcheck never goes green.** `make logs` shows all three. The MCP
healthcheck expects HTTP **401** — an unauthenticated request to a
bearer-protected endpoint should be rejected, and requiring exactly 401 proves an
authenticating MCP server is there rather than something else that took the port.

**Models changed and the container still serves the old ones.** The artefact is
baked in: `make rebuild`.

**`docker kill` does not trigger `restart: always`.** This surprises people (it
surprised me while testing this stack). Docker treats an explicit `docker kill`
or `docker stop` as an operator decision and suspends the restart policy until
the container is started again — so killing a container to "test self-healing"
proves nothing. To exercise the policy you have to make the process inside die:

```bash
docker exec longevity-backend python -c "import os,signal; os.kill(1, signal.SIGTERM)"
```

Verified: the backend exits, compose restarts it, and it is back to `healthy`
within ~12s with `RestartCount` incremented. (SIGKILL will not do it either — the
kernel ignores unhandled signals sent to PID 1 from inside its own namespace, and
the image has no `kill` binary anyway. SIGTERM works because uvicorn handles it.)
