"""Backward-compatible shims for database connections."""
from __future__ import annotations

from .db.connection import DEFAULT_DB_PATH, connection_scope, get_connection

__all__ = ["DEFAULT_DB_PATH", "connection_scope", "get_connection"]
