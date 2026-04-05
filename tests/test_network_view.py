from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pandas as pd

from dashboard.views import network_view


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _signal_bundle() -> SimpleNamespace:
    return SimpleNamespace(
        alert=SimpleNamespace(
            message="msg",
            severity="warning",
            title="title",
        ),
        metric_cards=lambda: [],
    )


def test_render_network_view_live_mode_renders_only_live_surface(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(network_view, "build_network_shell_signal_bundle", lambda _conn: _signal_bundle())
    monkeypatch.setattr(network_view, "render_signal_contract_notice", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(network_view, "render_kpi_strip", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(network_view, "render_alert_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(network_view, "tier_separator", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(network_view, "hero_section", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(network_view.st, "radio", lambda *args, **kwargs: "Live network")
    monkeypatch.setattr(network_view.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(network_view.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(network_view.st, "container", lambda: _DummyContext())
    monkeypatch.setattr(network_view.st, "columns", lambda spec: [_DummyContext() for _ in spec])
    monkeypatch.setattr(
        network_view,
        "render_distribution_analytics_surface",
        lambda **_kwargs: calls.append("live"),
    )
    monkeypatch.setattr(
        network_view,
        "render_route_maps_tab",
        lambda **_kwargs: calls.append("routes"),
    )

    network_view.render_network_view(
        conn=sqlite3.connect(":memory:"),
        filtered_df=pd.DataFrame(),
        mapping={},
        break_even_value=0.0,
        dataset_key="dataset",
        dataset_error=None,
    )

    assert calls == ["live"]


def test_render_network_view_route_mode_renders_only_route_surface(monkeypatch) -> None:
    route_calls: list[dict[str, object]] = []
    live_calls: list[str] = []

    monkeypatch.setattr(network_view, "build_network_shell_signal_bundle", lambda _conn: _signal_bundle())
    monkeypatch.setattr(network_view, "render_signal_contract_notice", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(network_view, "render_kpi_strip", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(network_view, "render_alert_banner", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(network_view, "tier_separator", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(network_view, "hero_section", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(network_view.st, "radio", lambda *args, **kwargs: "Heatmap")
    monkeypatch.setattr(network_view.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(network_view.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(network_view.st, "container", lambda: _DummyContext())
    monkeypatch.setattr(network_view.st, "columns", lambda spec: [_DummyContext() for _ in spec])
    monkeypatch.setattr(
        network_view,
        "render_distribution_analytics_surface",
        lambda **_kwargs: live_calls.append("live"),
    )
    monkeypatch.setattr(
        network_view,
        "render_route_maps_tab",
        lambda **kwargs: route_calls.append(kwargs),
    )

    network_view.render_network_view(
        conn=sqlite3.connect(":memory:"),
        filtered_df=pd.DataFrame(),
        mapping={},
        break_even_value=0.0,
        dataset_key="dataset",
        dataset_error=None,
    )

    assert live_calls == []
    assert len(route_calls) == 1
    assert route_calls[0]["show_title"] is False
    assert route_calls[0]["forced_mode"] == "Heatmap"
    assert route_calls[0]["network_host"] is True
