"""
db.py
-----
Database layer for persisting captured leads.

Supports two backends (configured via DB_BACKEND env var):
  - "csv"    : Append rows to data/leads.csv  (default)
  - "sqlite" : Store in data/leads.db via SQLite

Usage:
    from database.db import LeadDatabase
    db = LeadDatabase()
    db.save_lead(lead_object)
    df = db.get_all_leads()
"""

import csv
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "leads.csv"
SQLITE_PATH = DATA_DIR / "leads.db"

CSV_COLUMNS = [
    "name", "phone", "email", "budget",
    "preferred_location", "bhk_type", "buying_timeline",
    "notes", "timestamp",
]


# ---------------------------------------------------------------------------
# CSV backend
# ---------------------------------------------------------------------------

class _CSVBackend:
    """Append-only CSV store for leads."""

    def __init__(self, path: Path = CSV_PATH):
        self.path = path
        self._ensure_file()

    def _ensure_file(self):
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    def save(self, lead_dict: dict) -> None:
        row = {col: lead_dict.get(col, "") or "" for col in CSV_COLUMNS}
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow(row)
        logger.info(f"Lead saved to CSV: {self.path}")

    def get_all(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=CSV_COLUMNS)
        return pd.read_csv(self.path)


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

class _SQLiteBackend:
    """SQLite store for leads with full CRUD support."""

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS leads (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        name            TEXT,
        phone           TEXT,
        email           TEXT,
        budget          TEXT,
        preferred_location TEXT,
        bhk_type        TEXT,
        buying_timeline TEXT,
        notes           TEXT,
        timestamp       TEXT
    );
    """

    def __init__(self, path: Path = SQLITE_PATH):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(str(self.path))

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute(self.CREATE_TABLE_SQL)

    def save(self, lead_dict: dict) -> None:
        cols = CSV_COLUMNS  # same field names
        values = tuple(lead_dict.get(c, "") or "" for c in cols)
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        sql = f"INSERT INTO leads ({col_str}) VALUES ({placeholders})"

        with self._get_conn() as conn:
            conn.execute(sql, values)
        logger.info(f"Lead saved to SQLite: {self.path}")

    def get_all(self) -> pd.DataFrame:
        with self._get_conn() as conn:
            return pd.read_sql_query("SELECT * FROM leads ORDER BY id DESC", conn)


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

class LeadDatabase:
    """
    Unified interface for lead persistence.
    Set DB_BACKEND=sqlite in .env to use SQLite; default is CSV.
    """

    def __init__(self):
        backend = os.getenv("DB_BACKEND", "csv").lower()
        if backend == "sqlite":
            self._backend = _SQLiteBackend()
            logger.info("Using SQLite backend for leads.")
        else:
            self._backend = _CSVBackend()
            logger.info("Using CSV backend for leads.")

    def save_lead(self, lead) -> None:
        """Accept a Lead dataclass instance or dict and persist it."""
        if hasattr(lead, "to_dict"):
            lead_dict = lead.to_dict()
        else:
            lead_dict = dict(lead)
        self._backend.save(lead_dict)

    def get_all_leads(self) -> pd.DataFrame:
        """Return all leads as a Pandas DataFrame."""
        return self._backend.get_all()
