"""Migrate the schema and load the shipped SQLite fixture into Postgres.

`data/generate_db.py` stays the canonical definition of the mock clinic and is
deliberately untouched: regenerating the fixture and re-running this script is
the supported way to reset Postgres, so the two backends can never drift apart
in content.

Idempotent. Safe to run on every boot — it is wired as a one-shot compose service
the backend waits on:

  * schema      -> `alembic upgrade head` (no-op once at head)
  * demographics/biomarkers -> skipped entirely if already populated
  * risks       -> only the SEEDED history is copied (inputs_hash IS NULL).
                   Live computed rows are never re-imported, so a reseed cannot
                   resurrect deleted results or duplicate the trend.

Run:
    uv run python scripts/seed_postgres.py
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from sqlalchemy import func, insert, select  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from backend.app.core.config import settings  # noqa: E402
from backend.app.db.models import biomarkers, demographics, risks  # noqa: E402

SQLITE_PATH = REPO_ROOT / "data" / "patient_db.db"


def _read_fixture() -> dict[str, list[dict]]:
    if not SQLITE_PATH.exists():
        raise FileNotFoundError(
            f"{SQLITE_PATH} is missing — run `uv run python data/generate_db.py` first"
        )
    con = sqlite3.connect(SQLITE_PATH)
    con.row_factory = sqlite3.Row
    try:
        return {
            "demographics": [dict(r) for r in con.execute("SELECT * FROM demographics")],
            "biomarkers": [dict(r) for r in con.execute("SELECT * FROM biomarkers")],
            # Seeded history only: rows the fixture ships with, which carry no
            # inputs_json. Anything computed live belongs to whichever database
            # produced it.
            "risks": [
                dict(r)
                for r in con.execute(
                    "SELECT * FROM risks WHERE inputs_json IS NULL ORDER BY id"
                )
            ],
        }
    finally:
        con.close()


def run_migrations() -> None:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    command.upgrade(config, "head")


async def load() -> None:
    fixture = _read_fixture()
    engine = create_async_engine(settings.postgres_dsn, pool_pre_ping=True)

    async with engine.begin() as conn:
        existing = await conn.scalar(select(func.count()).select_from(demographics))
        if existing:
            print(f"Postgres already seeded ({existing} patients) — nothing to do.")
            await engine.dispose()
            return

        await conn.execute(insert(demographics), fixture["demographics"])
        # Let Postgres assign ids rather than importing SQLite's, so the
        # sequences stay consistent with future inserts.
        await conn.execute(
            insert(biomarkers),
            [{k: v for k, v in row.items() if k != "id"} for row in fixture["biomarkers"]],
        )
        await conn.execute(
            insert(risks),
            [
                {k: v for k, v in row.items() if k != "id"} | {"inputs_hash": None}
                for row in fixture["risks"]
            ],
        )

    async with engine.connect() as conn:
        counts = {
            "demographics": await conn.scalar(select(func.count()).select_from(demographics)),
            "biomarkers": await conn.scalar(select(func.count()).select_from(biomarkers)),
            "risks": await conn.scalar(select(func.count()).select_from(risks)),
        }
    await engine.dispose()

    print(f"Seeded Postgres from {SQLITE_PATH.name}:")
    for table, count in counts.items():
        print(f"  {table:14} {count} rows")


def main() -> int:
    print(f"Target: {settings.postgres_dsn.rsplit('@', 1)[-1]}")
    run_migrations()
    asyncio.run(load())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
