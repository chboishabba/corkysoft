"""Streamlit entrypoint for the price distribution dashboard."""
from __future__ import annotations

import streamlit as st

from dashboard.app import render_price_distribution_dashboard


def main() -> None:
    """Configure the Streamlit page and render the dashboard."""
    st.set_page_config(
        page_title="Price distribution by corridor",
        layout="wide",
    )
    render_price_distribution_dashboard()


if __name__ == "__main__":
    main()
