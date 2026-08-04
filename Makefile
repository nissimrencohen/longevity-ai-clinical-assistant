# Convenience targets. Each long-running service wants its own terminal.
# Ports: backend 8001 · MLflow 5001 · MCP 9000 · LibreChat 3080 (Docker).

.PHONY: help install data db models register backend mcp mlflow test test-unit \
        eval eval-agent eval-all lint up down logs ps rebuild

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv and install dependencies
	uv sync

data: db models  ## Regenerate the mock DB and the models

db:  ## Regenerate data/patient_db.db
	uv run python data/generate_db.py

models:  ## Regenerate models/*.pkl
	uv run python models/generate_models.py

backend:  ## Run the FastAPI backend on :8001 (implement the endpoints first)
	uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8001 --reload

mcp:  ## Run the FastMCP server on 0.0.0.0:9000
	uv run python mcp-server/server.py

register:  ## Register the RiskRouter pyfunc into models/mlflow_risk_router
	uv run python models/register_router.py

mlflow: register  ## Serve the risk router on :5001 (registers it first)
	uv run mlflow models serve -m models/mlflow_risk_router -p 5001 --host 127.0.0.1 --env-manager local

test:  ## Run all tests (MLflow integration tests auto-skip if :5001 is down)
	uv run pytest

test-unit:  ## Run only the hermetic tests — no services required
	uv run pytest -m "not integration"

eval:  ## Run the deterministic eval tier (no API key needed) — the regression gate
	uv run python evals/harness.py --tier a

eval-agent:  ## Run the agent-in-the-loop eval tier (needs OPENROUTER_KEY)
	uv run python evals/harness.py --tier b --repeats 3

eval-all:  ## Run both eval tiers
	uv run python evals/harness.py --tier both --repeats 3

lint:  ## Lint with ruff
	uv run ruff check .

# --- Containerised stack (recommended) --------------------------------------
# The three services run on a private network and restart automatically. Only
# the MCP port is published, for the eval harness and LibreChat.

up:  ## Start the whole stack in Docker
	docker compose up -d --wait

up-debug:  ## Start the stack with backend + MLflow also published to the host
	docker compose -f docker-compose.yml -f docker-compose.debug.yml up -d --wait

down:  ## Stop the stack
	docker compose down

rebuild:  ## Rebuild images and restart (after changing code or models)
	docker compose up -d --build --wait

ps:  ## Show container status and health
	docker compose ps

logs:  ## Tail logs from all services
	docker compose logs -f --tail=100
