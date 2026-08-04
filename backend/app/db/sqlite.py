"""Thin async SQLite access helper.

The database is tiny, but the endpoints are ``async`` — so use ``aiosqlite``,
NOT the stdlib ``sqlite3`` (whose blocking calls would stall the event loop).

This is the connection primitive; the queries live in
``app/db/store.py::SqliteStore``, which is one of the two interchangeable
backends behind the ``ClinicalStore`` protocol.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from ..core.config import settings


@asynccontextmanager
async def open_db(db_path: Path | str | None = None) -> AsyncIterator[aiosqlite.Connection]:
    """Yield an aiosqlite connection with ``Row`` access (dict-like rows).

    ``db_path`` defaults to ``settings.patient_db_path``; tests override it to
    point at a disposable copy so the shipped fixture stays pristine.

    Usage:
        async with open_db() as db:
            async with db.execute("SELECT ... WHERE patient_id = ?", (pid,)) as cur:
                row = await cur.fetchone()
    """
    conn = await aiosqlite.connect(db_path or settings.patient_db_path)
    conn.row_factory = aiosqlite.Row
    try:
        yield conn
    finally:
        await conn.close()
