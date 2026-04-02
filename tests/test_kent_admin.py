from __future__ import annotations

import dashboard.components.kent as kent


def test_kent_admin_write_guard() -> None:
    assert kent._kent_admin_write_enabled("system_rollout_admin")
    assert not kent._kent_admin_write_enabled("estimator")
    assert not kent._kent_admin_write_enabled("dispatcher")
    assert not kent._kent_admin_write_enabled(None)


def test_kent_admin_write_roles_are_limited() -> None:
    assert kent.KENT_ADMIN_WRITE_ROLES == frozenset({"system_rollout_admin"})
