from __future__ import annotations

import inspect

from dashboard.views.admin_view import render_admin_view
from dashboard.views.network_view import render_network_view
from dashboard.views.operations_view import render_operations_view
from dashboard.views.pricing_intelligence_view import render_pricing_intelligence_view
from dashboard.views.quote_view import render_quote_view


def test_shell_views_render_placeholder_governance_notice() -> None:
    for render in (
        render_quote_view,
        render_pricing_intelligence_view,
        render_network_view,
        render_operations_view,
        render_admin_view,
    ):
        source = inspect.getsource(render)
        assert "render_signal_contract_notice(" in source

    network_source = inspect.getsource(render_network_view)
    assert "build_network_shell_signal_bundle(" in network_source
    assert 'with st.expander("🗺️ Historic Route Maps"' not in network_source
    assert network_source.count("render_route_maps_tab(") == 1
    assert "network_host=True" in network_source
    operations_source = inspect.getsource(render_operations_view)
    assert "build_operations_shell_signal_bundle(" in operations_source
    quote_source = inspect.getsource(render_quote_view)
    assert "build_quote_shell_signal_bundle(" in quote_source
    pricing_source = inspect.getsource(render_pricing_intelligence_view)
    assert "build_pricing_shell_signal_bundle(" in pricing_source
    admin_source = inspect.getsource(render_admin_view)
    assert "build_admin_shell_signal_bundle(" in admin_source
