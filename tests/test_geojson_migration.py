import sqlite3

from analytics.db import ensure_historical_job_routes_table, migrate_geojson_to_routes


def _historical_job_columns(conn: sqlite3.Connection) -> list[str]:
    return [row[1] for row in conn.execute("PRAGMA table_info(historical_jobs)")]


def test_migrate_geojson_moves_rows_and_drops_column():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE historical_jobs (
            id INTEGER PRIMARY KEY,
            job_date TEXT,
            route_geojson TEXT,
            imported_at TEXT,
            updated_at TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO historical_jobs VALUES (?, ?, ?, ?, ?)",
        [
            (1, "2024-01-01", "{\"type\":\"LineString\"}", "2024-01-02", "2024-01-03"),
            (2, "2024-01-05", "", "2024-01-06", "2024-01-07"),
            (3, "2024-02-01", None, "2024-02-02", None),
        ],
    )
    conn.commit()

    migrate_geojson_to_routes(conn)

    columns = _historical_job_columns(conn)
    assert "route_geojson" not in columns

    rows = conn.execute(
        "SELECT historical_job_id, geojson, created_at, updated_at FROM historical_job_routes"
    ).fetchall()
    assert rows == [(1, '{"type":"LineString"}', "2024-01-03", "2024-01-03")]


def test_migration_is_idempotent():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE historical_jobs (
            id INTEGER PRIMARY KEY,
            route_geojson TEXT,
            imported_at TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO historical_jobs VALUES (1, '{\"type\":\"LineString\"}', '2024-03-01')"
    )
    conn.commit()

    migrate_geojson_to_routes(conn)
    migrate_geojson_to_routes(conn)

    ensure_historical_job_routes_table(conn)
    rows = conn.execute("SELECT historical_job_id, geojson FROM historical_job_routes").fetchall()
    assert rows == [(1, '{"type":"LineString"}')]


def test_migration_noop_when_table_missing():
    conn = sqlite3.connect(":memory:")

    # Should not raise even if the table is absent.
    migrate_geojson_to_routes(conn)

    # The helper should still allow creating the table manually if needed.
    ensure_historical_job_routes_table(conn)
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    assert "historical_job_routes" in tables
