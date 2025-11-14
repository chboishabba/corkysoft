"""Legacy entry point for the price distribution dashboard."""
from __future__ import annotations

from dashboard.app import (
    main as _app_main,
    render_price_distribution_dashboard as _render_price_distribution_dashboard,
)

__all__ = ["main", "render_price_distribution_dashboard"]

render_price_distribution_dashboard = _render_price_distribution_dashboard


def main() -> None:
    """Run the Streamlit price distribution dashboard."""

    _app_main()


if __name__ == "__main__":  # pragma: no cover - convenience for manual execution
    main()
