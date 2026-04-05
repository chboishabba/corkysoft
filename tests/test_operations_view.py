from dashboard.views.operations_view import _display_operations_primary_tabs


def test_display_operations_primary_tabs_keeps_canonical_order_when_key_supported() -> None:
    labels = ["Dispatch", "Planner", "Operations Diary"]

    display_labels = _display_operations_primary_tabs(
        labels,
        "Planner",
        can_assign_tab_key=True,
    )

    assert display_labels == labels


def test_display_operations_primary_tabs_promotes_requested_tab_when_key_unsupported() -> None:
    labels = ["Dispatch", "Planner", "Operations Diary"]

    display_labels = _display_operations_primary_tabs(
        labels,
        "Operations Diary",
        can_assign_tab_key=False,
    )

    assert display_labels == ["Operations Diary", "Dispatch", "Planner"]
