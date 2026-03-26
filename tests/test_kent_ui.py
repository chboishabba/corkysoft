from dashboard.components.kent import _kent_admin_write_enabled


def test_kent_admin_write_enabled_for_system_rollout_admin() -> None:
    assert _kent_admin_write_enabled("system_rollout_admin") is True


def test_kent_admin_write_disabled_for_non_admin_roles() -> None:
    assert _kent_admin_write_enabled("dispatcher") is False
    assert _kent_admin_write_enabled("operations_manager") is False
    assert _kent_admin_write_enabled(None) is False
