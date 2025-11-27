import sqlite3

from analytics.db import (
    ensure_global_parameters_table,
    get_parameter_value,
    set_parameter_value,
)


def test_set_and_get_parameter_round_trip() -> None:
    conn = sqlite3.connect(":memory:")

    ensure_global_parameters_table(conn)
    set_parameter_value(conn, "test_key", 42.5, "example value")

    assert get_parameter_value(conn, "test_key") == 42.5
    assert get_parameter_value(conn, "missing", default=1.0) == 1.0
