import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dashboard.components.kent as kent_module
from dashboard.components.kent import _kent_admin_write_enabled, render_kent_admin_tab


def test_kent_admin_write_enabled_for_system_rollout_admin() -> None:
    assert _kent_admin_write_enabled("system_rollout_admin") is True


def test_kent_admin_write_disabled_for_non_admin_roles() -> None:
    assert _kent_admin_write_enabled("dispatcher") is False
    assert _kent_admin_write_enabled("operations_manager") is False
    assert _kent_admin_write_enabled(None) is False


def test_render_kent_admin_tab_passes_role_key_by_keyword(monkeypatch) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    calls: list[tuple[sqlite3.Connection, str]] = []

    def fake_render_dashboard_user_admin(connection: sqlite3.Connection, *, current_role_key: str) -> None:
        calls.append((connection, current_role_key))

    monkeypatch.setattr(kent_module.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(kent_module.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(kent_module.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(kent_module.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(kent_module.st, "dataframe", lambda *args, **kwargs: None)

    class _DummyForm:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyColumn:
        def selectbox(self, *args, **kwargs):
            return "ABS_ONLY"

        def number_input(self, *args, **kwargs):
            return 0.0

        def text_input(self, *args, **kwargs):
            return ""

        def checkbox(self, *args, **kwargs):
            return True

        def metric(self, *args, **kwargs):
            return None

    monkeypatch.setattr(kent_module.st, "form", lambda *args, **kwargs: _DummyForm())
    monkeypatch.setattr(kent_module.st, "columns", lambda n: [_DummyColumn() for _ in range(n)])
    monkeypatch.setattr(kent_module.st, "form_submit_button", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        kent_module,
        "get_kent_tender_policy_config",
        lambda connection: {
            "ruleMode": "ABS_ONLY",
            "absoluteMarginThreshold": 0.0,
            "marginPercentThreshold": 0.0,
            "lossAlertFloor": 0.0,
            "requireReasonForOverride": False,
        },
    )
    monkeypatch.setattr(kent_module, "list_kent_override_reason_codes", lambda connection: [])
    monkeypatch.setattr(kent_module, "list_prioritized_tenders", lambda connection, **kwargs: [])
    monkeypatch.setattr(kent_module, "list_kent_tender_override_history", lambda connection, **kwargs: [])

    render_kent_admin_tab(
        conn,
        current_role_key="system_rollout_admin",
        rerun_app=lambda: None,
        render_dashboard_user_admin=fake_render_dashboard_user_admin,
    )

    assert calls == [(conn, "system_rollout_admin")]


def test_render_kent_admin_tab_disables_write_controls_for_non_admin(monkeypatch) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    disabled_values: list[bool] = []
    submit_disabled_values: list[bool] = []

    def _record_disabled(kwargs):
        disabled_values.append(bool(kwargs.get("disabled", False)))

    monkeypatch.setattr(kent_module.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(kent_module.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(kent_module.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(kent_module.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(kent_module.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(kent_module.st, "success", lambda *args, **kwargs: None)
    monkeypatch.setattr(kent_module.st, "error", lambda *args, **kwargs: None)

    class _DummyForm:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyColumn:
        def selectbox(self, *args, **kwargs):
            _record_disabled(kwargs)
            return "ABS_ONLY"

        def number_input(self, *args, **kwargs):
            _record_disabled(kwargs)
            return 0.0

        def text_input(self, *args, **kwargs):
            _record_disabled(kwargs)
            return ""

        def checkbox(self, *args, **kwargs):
            _record_disabled(kwargs)
            return True

        def metric(self, *args, **kwargs):
            return None

    monkeypatch.setattr(kent_module.st, "form", lambda *args, **kwargs: _DummyForm())
    monkeypatch.setattr(kent_module.st, "columns", lambda n: [_DummyColumn() for _ in range(n)])

    def _fake_form_submit_button(*args, **kwargs):
        submit_disabled_values.append(bool(kwargs.get("disabled", False)))
        return False

    monkeypatch.setattr(kent_module.st, "form_submit_button", _fake_form_submit_button)
    monkeypatch.setattr(
        kent_module,
        "get_kent_tender_policy_config",
        lambda connection: {
            "ruleMode": "ABS_ONLY",
            "absoluteMarginThreshold": 0.0,
            "marginPercentThreshold": 0.0,
            "lossAlertFloor": 0.0,
            "requireReasonForOverride": False,
        },
    )
    monkeypatch.setattr(kent_module, "list_kent_override_reason_codes", lambda connection: [])
    monkeypatch.setattr(kent_module, "list_prioritized_tenders", lambda connection, **kwargs: [])
    monkeypatch.setattr(kent_module, "list_kent_tender_override_history", lambda connection, **kwargs: [])

    render_kent_admin_tab(
        conn,
        current_role_key="dispatcher",
        rerun_app=lambda: None,
        render_dashboard_user_admin=None,
    )

    assert disabled_values
    assert all(disabled_values)
    assert submit_disabled_values
    assert all(submit_disabled_values)
