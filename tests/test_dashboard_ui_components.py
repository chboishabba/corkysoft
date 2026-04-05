from __future__ import annotations

from dashboard.components.alert_banner import render_alert_banner
from dashboard.components.data_provenance import render_signal_contract_notice
from dashboard.components.kpi_strip import render_kpi_strip
from dashboard.shell_signals import ShellAlertSignal, ShellSignalBundle


def test_render_kpi_strip_escapes_metric_content(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_markdown(body: str, *, unsafe_allow_html: bool = False) -> None:
        captured["body"] = body
        captured["unsafe"] = unsafe_allow_html

    monkeypatch.setattr("dashboard.components.kpi_strip.st.markdown", _fake_markdown)

    render_kpi_strip(
        [
            {
                "label": '<b>Win Rate</b>',
                "value": '<script>alert("x")</script>',
                "delta": '<img src=x onerror=alert(1)>',
                "direction": "up",
            }
        ]
    )

    body = str(captured["body"])
    assert captured["unsafe"] is True
    assert "&lt;b&gt;Win Rate&lt;/b&gt;" in body
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in body
    assert "&lt;img src=x onerror=alert(1)&gt;" in body
    assert '<script>alert("x")</script>' not in body


def test_render_alert_banner_escapes_title_and_message(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_markdown(body: str, *, unsafe_allow_html: bool = False) -> None:
        captured["body"] = body
        captured["unsafe"] = unsafe_allow_html

    monkeypatch.setattr("dashboard.components.alert_banner.st.markdown", _fake_markdown)

    render_alert_banner(
        '<script>alert("msg")</script>',
        severity="critical",
        title='<b>Alert</b>',
    )

    body = str(captured["body"])
    assert captured["unsafe"] is True
    assert "&lt;b&gt;Alert&lt;/b&gt;" in body
    assert "&lt;script&gt;alert(&quot;msg&quot;)&lt;/script&gt;" in body
    assert '<script>alert("msg")</script>' not in body


def test_render_signal_contract_notice_for_scaffold_state(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_chip(text: str, icon: str = "🔗") -> None:
        captured["text"] = text
        captured["icon"] = icon

    monkeypatch.setattr("dashboard.components.data_provenance.provenance_chip", _fake_chip)

    bundle = ShellSignalBundle(
        scope_label="Quote",
        metrics=[],
        alert=ShellAlertSignal(
            signal_id="alert",
            title="Title",
            message="Body",
            severity="warning",
            source="Quote shell scaffold",
            owner="Commercial operations",
            refresh_cadence="manual placeholder",
            stale_threshold="not applicable until sourced",
            freshness_state="scaffold",
            decision_grade="placeholder",
            fallback_behavior="render explicit placeholder notice instead of implying live truth",
        ),
        owner="Commercial operations",
        source="Quote shell scaffold",
        refresh_cadence="manual placeholder",
        stale_threshold="not applicable until sourced",
        freshness_state="scaffold",
        decision_grade="placeholder",
        fallback_behavior="render governance notice and avoid decision-grade claims",
    )

    render_signal_contract_notice(bundle)

    body = str(captured["text"])
    assert body == "Quote · scaffold values · Commercial operations · non-decision-grade"
    assert captured["icon"] == "🏗️"


def test_render_signal_contract_notice_for_stale_state(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_chip(text: str, icon: str = "🔗") -> None:
        captured["text"] = text
        captured["icon"] = icon

    monkeypatch.setattr("dashboard.components.data_provenance.provenance_chip", _fake_chip)

    bundle = ShellSignalBundle(
        scope_label="Network",
        metrics=[],
        alert=ShellAlertSignal(
            signal_id="alert",
            title="Title",
            message="Body",
            severity="info",
            source="Telemetry",
            owner="Network control",
            refresh_cadence="5m",
            stale_threshold="15m",
            freshness_state="stale",
            decision_grade="advisory",
            fallback_behavior="downgrade to stale banner",
        ),
        owner="Network control",
        source="Telemetry",
        refresh_cadence="5m",
        stale_threshold="15m",
        freshness_state="stale",
        decision_grade="advisory",
        fallback_behavior="downgrade to stale banner",
    )

    render_signal_contract_notice(bundle)

    body = str(captured["text"])
    assert body == "Network · stale · Network control · fallback active"
    assert captured["icon"] == "⏳"
