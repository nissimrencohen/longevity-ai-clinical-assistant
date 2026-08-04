# One image, three services.
#
# The backend, the MCP server and the MLflow model server share the same
# dependency set (they come from one uv.lock), so building three images would
# mean three copies of scikit-learn, pandas and mlflow for no benefit. They
# differ only in the command compose gives them.
#
# The MLflow router is REGISTERED AT BUILD TIME. models/mlflow_risk_router is
# generated from the committed pickles and is gitignored, so a clean clone has
# no artefact to serve; baking it in is what makes `docker compose up` work on
# a fresh checkout. It also means models/ must NOT be bind-mounted at runtime,
# or the host directory would shadow the artefact.

FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first, so editing source does not invalidate the (slow) install
# layer. --frozen refuses to silently update uv.lock inside a build.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

# Register the pyfunc router that serves all five models and returns
# predict_proba rather than class labels. Round-trips the P004 CKD anchor as it
# goes, so a build that produces a broken model fails here rather than at 3am.
RUN python models/register_router.py

# Non-root. The bind-mounted data/ directory stays writable because the risks
# table is an append log the backend writes to on every request.
RUN useradd --create-home --uid 10001 clinic \
    && chown -R clinic:clinic /app
USER clinic

EXPOSE 8001 5001 9000
