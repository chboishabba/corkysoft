"""Corkysoft design system — CSS injection for the guided workflow UI.

Call ``inject_css()`` once at the top of the Streamlit render tree
to apply the visual design system across all views.
"""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* ═══════════════════════════ GOOGLE FONTS ═══════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

/* ═══════════════════════════ DESIGN TOKENS ═══════════════════════════ */
:root {
    /* ── Surfaces ── */
    --ck-bg-primary: Canvas;
    --ck-bg-primary-alt: color-mix(in srgb, Canvas 92%, AccentColor 8%);
    --ck-bg-card: color-mix(in srgb, Canvas 90%, CanvasText 10%);
    --ck-bg-card-solid: color-mix(in srgb, Canvas 96%, CanvasText 4%);
    --ck-bg-card-alt: color-mix(in srgb, Canvas 84%, CanvasText 16%);
    --ck-bg-hero: linear-gradient(
        135deg,
        color-mix(in srgb, Canvas 96%, AccentColor 4%) 0%,
        color-mix(in srgb, Canvas 88%, AccentColor 12%) 100%
    );
    --ck-bg-sidebar: color-mix(in srgb, Canvas 94%, AccentColor 6%);
    --ck-bg-sidebar-alt: color-mix(in srgb, Canvas 86%, AccentColor 14%);
    --ck-border: color-mix(in srgb, Canvas 78%, CanvasText 22%);
    --ck-border-accent: color-mix(in srgb, Canvas 68%, AccentColor 32%);
    --ck-border-focus: #c97f00;

    /* ── Elevation ── */
    --ck-shadow-xs: 0 1px 2px rgba(30,40,80,0.04);
    --ck-shadow-sm: 0 1px 4px rgba(30,40,80,0.06), 0 1px 2px rgba(30,40,80,0.04);
    --ck-shadow-md: 0 4px 12px rgba(30,40,80,0.08), 0 1px 3px rgba(30,40,80,0.05);
    --ck-shadow-lg: 0 8px 24px rgba(30,40,80,0.10), 0 2px 6px rgba(30,40,80,0.06);
    --ck-shadow-glow-red: 0 0 16px rgba(224,49,49,0.18), 0 0 4px rgba(224,49,49,0.12);
    --ck-shadow-glow-amber: 0 0 12px rgba(202,121,0,0.14);

    /* ── Typography ── */
    --ck-font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --ck-text-primary: CanvasText;
    --ck-text-secondary: color-mix(in srgb, CanvasText 72%, Canvas 28%);
    --ck-text-muted: color-mix(in srgb, CanvasText 48%, Canvas 52%);
    --ck-text-on-sidebar: CanvasText;

    /* ── Accent palette ── */
    --ck-accent: #c97f00;
    --ck-accent-hover: #a86400;
    --ck-accent-light: color-mix(in srgb, Canvas 88%, AccentColor 12%);
    --ck-accent-gradient: linear-gradient(135deg, #c97f00 0%, #b56d00 50%, #a05d00 100%);

    /* ── Semantic colors ── */
    --ck-green: #0f9d58;
    --ck-green-bg: color-mix(in srgb, Canvas 82%, #0f9d58 18%);
    --ck-green-border: rgba(15,157,88,0.25);
    --ck-red: #e03131;
    --ck-red-bg: color-mix(in srgb, Canvas 82%, #e03131 18%);
    --ck-red-border: rgba(224,49,49,0.25);
    --ck-amber: #d97706;
    --ck-amber-bg: color-mix(in srgb, Canvas 82%, #d97706 18%);
    --ck-amber-border: rgba(217,119,6,0.25);
    --ck-blue: #475569;
    --ck-blue-bg: color-mix(in srgb, Canvas 82%, #475569 18%);
    --ck-blue-border: rgba(71,85,105,0.20);

    /* ── Radii ── */
    --ck-radius: 10px;
    --ck-radius-sm: 6px;
    --ck-radius-lg: 14px;
    --ck-radius-pill: 100px;

    /* ── Transitions ── */
    --ck-ease: cubic-bezier(0.4, 0, 0.2, 1);
    --ck-transition-fast: 150ms var(--ck-ease);
    --ck-transition: 250ms var(--ck-ease);
}

/* ═══════════════════════════ GLOBAL RESET ═══════════════════════════ */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: var(--ck-font) !important;
}
.stApp {
    background: var(--ck-bg-primary) !important;
    color: var(--ck-text-primary) !important;
}
/* Subtle background texture */
[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(180deg, var(--ck-bg-primary) 0%, var(--ck-bg-primary-alt) 100%) !important;
}

/* ═══════════════════════════ TYPOGRAPHY ═══════════════════════════ */
.ck-section-title {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: var(--ck-text-primary) !important;
    margin-bottom: 0.15rem !important;
    letter-spacing: -0.025em;
    line-height: 1.3;
}
.ck-section-subtitle {
    font-size: 0.88rem;
    color: var(--ck-text-muted);
    margin-bottom: 1.25rem;
    line-height: 1.45;
}

/* ═══════════════════════════ KPI STRIP ═══════════════════════════ */
.ck-kpi-strip {
    display: flex;
    gap: 14px;
    margin-bottom: 20px;
}
.ck-kpi-card {
    flex: 1;
    background: var(--ck-bg-card);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--ck-border);
    border-radius: var(--ck-radius);
    padding: 18px 22px;
    box-shadow: var(--ck-shadow-sm);
    border-left: 4px solid var(--ck-border-accent);
    transition: box-shadow var(--ck-transition), transform var(--ck-transition-fast);
    position: relative;
    overflow: hidden;
}
.ck-kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(201,127,0,0.04) 0%, transparent 70%);
    transform: translate(20px, -20px);
}
.ck-kpi-card:hover {
    box-shadow: var(--ck-shadow-md);
    transform: translateY(-2px);
}
.ck-kpi-card.delta-up { border-left-color: var(--ck-green); }
.ck-kpi-card.delta-down { border-left-color: var(--ck-red); }
.ck-kpi-card.delta-neutral { border-left-color: var(--ck-accent); }

.ck-kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--ck-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}
.ck-kpi-value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--ck-text-primary);
    line-height: 1.1;
    letter-spacing: -0.03em;
}
.ck-kpi-delta {
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 6px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: var(--ck-radius-pill);
}
.ck-kpi-delta.up {
    color: var(--ck-green);
    background: var(--ck-green-bg);
}
.ck-kpi-delta.down {
    color: var(--ck-red);
    background: var(--ck-red-bg);
}
.ck-kpi-delta.neutral {
    color: var(--ck-text-muted);
    background: var(--ck-bg-card-alt);
}

/* ═══════════════════════════ ALERT BANNERS ═══════════════════════════ */
.ck-alert {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 16px 20px;
    border-radius: var(--ck-radius);
    margin-bottom: 18px;
    font-size: 0.9rem;
    line-height: 1.5;
    border-left: 5px solid transparent;
    transition: box-shadow var(--ck-transition);
}
.ck-alert-icon {
    font-size: 1.3rem;
    flex-shrink: 0;
    margin-top: 1px;
}
.ck-alert-body {
    flex: 1;
    min-width: 0;
}
.ck-alert-title {
    font-weight: 700;
    font-size: 0.92rem;
    margin-bottom: 3px;
    letter-spacing: -0.01em;
}

/* Critical — red, pulsing glow */
.ck-alert.critical {
    background: var(--ck-red-bg);
    border-left-color: var(--ck-red);
    color: #7f1d1d;
    box-shadow: var(--ck-shadow-glow-red);
}
.ck-alert.critical .ck-alert-icon { color: var(--ck-red); }

@keyframes ck-pulse-border {
    0%, 100% { border-left-color: var(--ck-red); box-shadow: var(--ck-shadow-glow-red); }
    50% { border-left-color: #f87171; box-shadow: 0 0 24px rgba(248,113,113,0.25), 0 0 6px rgba(224,49,49,0.15); }
}
.ck-alert.critical {
    animation: ck-pulse-border 2.5s ease-in-out infinite;
}

/* Warning — amber */
.ck-alert.warning {
    background: var(--ck-amber-bg);
    border-left-color: var(--ck-amber);
    color: #78350f;
    box-shadow: var(--ck-shadow-glow-amber);
}
.ck-alert.warning .ck-alert-icon { color: var(--ck-amber); }

/* Info — blue */
.ck-alert.info {
    background: var(--ck-blue-bg);
    border-left-color: var(--ck-blue);
    color: color-mix(in srgb, CanvasText 68%, Canvas 32%);
    box-shadow: var(--ck-shadow-glow-amber);
}
.ck-alert.info .ck-alert-icon { color: var(--ck-blue); }

/* ═══════════════════════════ PROVENANCE CHIP ═══════════════════════════ */
.ck-provenance {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--ck-text-muted);
    background: var(--ck-bg-card-alt);
    border: 1px solid var(--ck-border);
    border-radius: var(--ck-radius-pill);
    padding: 4px 14px;
    margin-bottom: 14px;
    letter-spacing: 0.01em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
}
.ck-provenance-icon {
    font-size: 0.8rem;
    opacity: 0.6;
}

/* ═══════════════════════════ HERO SECTION ═══════════════════════════ */
.ck-hero-section {
    background: var(--ck-bg-hero);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--ck-border);
    border-radius: var(--ck-radius-lg);
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: var(--ck-shadow-md);
    position: relative;
    overflow: hidden;
}
.ck-hero-section::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--ck-accent-gradient);
    border-radius: var(--ck-radius-lg) var(--ck-radius-lg) 0 0;
}

/* ═══════════════════════════ SECTION CARDS ═══════════════════════════ */
.ck-card {
    background: var(--ck-bg-card);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--ck-border);
    border-radius: var(--ck-radius);
    padding: 22px 26px;
    margin-bottom: 20px;
    box-shadow: var(--ck-shadow-sm);
    transition: box-shadow var(--ck-transition);
}
.ck-card:hover {
    box-shadow: var(--ck-shadow-md);
}
.ck-card-muted {
    background: var(--ck-bg-card-alt);
    border: 1px solid var(--ck-border);
    border-radius: var(--ck-radius);
    padding: 16px 20px;
    margin-bottom: 16px;
}

/* ═══════════════════════════ ACTION BAR ═══════════════════════════ */
.ck-action-bar {
    background: linear-gradient(135deg, #1f2937 0%, #111827 60%, #0f172a 100%);
    border-radius: var(--ck-radius);
    padding: 14px 24px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin-top: 8px;
    margin-bottom: 10px;
    box-shadow: var(--ck-shadow-lg);
    position: relative;
    overflow: hidden;
}
.ck-action-bar::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--ck-accent-gradient);
}
.ck-action-bar .ck-action-label {
    color: rgba(255,255,255,0.55);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-right: auto;
}

/* ═══════════════════════════ STATUS PILLS ═══════════════════════════ */
.ck-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.73rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: var(--ck-radius-pill);
    letter-spacing: 0.02em;
    white-space: nowrap;
}
.ck-status-pill.green {
    background: var(--ck-green-bg);
    color: var(--ck-green);
    border: 1px solid var(--ck-green-border);
}
.ck-status-pill.red {
    background: var(--ck-red-bg);
    color: var(--ck-red);
    border: 1px solid var(--ck-red-border);
}
.ck-status-pill.amber {
    background: var(--ck-amber-bg);
    color: var(--ck-amber);
    border: 1px solid var(--ck-amber-border);
}
.ck-status-pill.blue {
    background: var(--ck-blue-bg);
    color: var(--ck-blue);
    border: 1px solid var(--ck-blue-border);
}
.ck-status-pill.neutral {
    background: var(--ck-bg-card-alt);
    color: var(--ck-text-secondary);
    border: 1px solid var(--ck-border);
}

/* ═══════════════════════════ TIER SEPARATOR ═══════════════════════════ */
.ck-tier-sep {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--ck-border-accent) 30%, var(--ck-border-accent) 70%, transparent 100%);
    margin: 28px 0;
    opacity: 0.6;
}

/* ═══════════════════════════ STREAMLIT OVERRIDES ═══════════════════════ */

/* ── Global font ── */
.stMarkdown, .stText, p, label, .stCaption,
div[data-testid="stMetricLabel"],
div[data-testid="stMetricValue"],
div[data-testid="stMetricDelta"] {
    font-family: var(--ck-font) !important;
}

span[data-testid="stIconMaterial"],
.material-symbols-rounded,
.material-symbols-outlined {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-feature-settings: "liga" 1 !important;
    -webkit-font-feature-settings: "liga" 1 !important;
    font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 24 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    line-height: 1;
}

/* ── Form inputs — recessed, quiet appearance ── */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stMultiSelect"] > div > div,
div[data-testid="stDateInput"] input {
    border: 1px solid var(--ck-border) !important;
    border-radius: var(--ck-radius-sm) !important;
    background: var(--ck-bg-card-alt) !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.04) !important;
    font-family: var(--ck-font) !important;
    font-size: 0.9rem !important;
    transition: border-color var(--ck-transition-fast), box-shadow var(--ck-transition-fast) !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stDateInput"] input:focus {
    border-color: var(--ck-border-focus) !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.04), 0 0 0 3px rgba(201,127,0,0.12) !important;
    background: var(--ck-bg-card-solid) !important;
}

/* ── Slider styling ── */
div[data-testid="stSlider"] > div > div > div {
    font-family: var(--ck-font) !important;
}

/* ── Tab bar ── */
div[data-testid="stTabs"] > div[role="tablist"] {
    gap: 0 !important;
    border-bottom: 2px solid var(--ck-border) !important;
    padding: 0 4px !important;
}
div[data-testid="stTabs"] > div[role="tablist"] button {
    font-family: var(--ck-font) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 12px 22px !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px !important;
    color: var(--ck-text-muted) !important;
    transition: color var(--ck-transition-fast), border-color var(--ck-transition-fast), background var(--ck-transition-fast) !important;
    border-radius: var(--ck-radius-sm) var(--ck-radius-sm) 0 0 !important;
}
div[data-testid="stTabs"] > div[role="tablist"] button[aria-selected="true"] {
    color: var(--ck-accent) !important;
    border-bottom-color: var(--ck-accent) !important;
    font-weight: 700 !important;
    background: var(--ck-accent-light) !important;
}
div[data-testid="stTabs"] > div[role="tablist"] button:hover:not([aria-selected="true"]) {
    color: var(--ck-text-primary) !important;
    background: rgba(201,127,0,0.04) !important;
}

/* ── Expander — drawer feel ── */
div[data-testid="stExpander"] {
    border: 1px solid var(--ck-border) !important;
    border-radius: var(--ck-radius) !important;
    box-shadow: var(--ck-shadow-sm) !important;
    overflow: hidden !important;
    transition: box-shadow var(--ck-transition) !important;
    background: var(--ck-bg-card) !important;
}
div[data-testid="stExpander"]:hover {
    box-shadow: var(--ck-shadow-md) !important;
}
div[data-testid="stExpander"] summary {
    font-family: var(--ck-font) !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    color: var(--ck-text-secondary) !important;
    padding: 14px 20px !important;
    transition: color var(--ck-transition-fast) !important;
}
div[data-testid="stExpander"] summary:hover {
    color: var(--ck-text-primary) !important;
}

/* ── Metric override ── */
div[data-testid="stMetric"] {
    background: transparent !important;
}
div[data-testid="stMetricLabel"] > label {
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    color: var(--ck-text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
div[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
}

/* ── Primary button ── */
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stButton"] > button.st-emotion-cache-primary {
    background: var(--ck-accent-gradient) !important;
    border: none !important;
    border-radius: var(--ck-radius-sm) !important;
    font-family: var(--ck-font) !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 2px 6px rgba(201,127,0,0.3) !important;
    transition: transform var(--ck-transition-fast), box-shadow var(--ck-transition-fast) !important;
    padding: 8px 20px !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(201,127,0,0.35) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:active {
    transform: translateY(0) !important;
}

/* ── Secondary button ── */
div[data-testid="stButton"] > button:not([kind="primary"]) {
    border: 1px solid var(--ck-border-accent) !important;
    border-radius: var(--ck-radius-sm) !important;
    color: var(--ck-text-secondary) !important;
    font-family: var(--ck-font) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    background: var(--ck-bg-card-solid) !important;
    transition: border-color var(--ck-transition-fast), color var(--ck-transition-fast), box-shadow var(--ck-transition-fast) !important;
    padding: 8px 20px !important;
}
div[data-testid="stButton"] > button:not([kind="primary"]):hover {
    border-color: var(--ck-accent) !important;
    color: var(--ck-accent) !important;
    box-shadow: var(--ck-shadow-sm) !important;
}

/* ── Tables — refined styling ── */
div[data-testid="stDataFrame"] table,
.stDataFrame table,
table {
    font-family: var(--ck-font) !important;
    font-size: 0.84rem !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
}
div[data-testid="stDataFrame"] th,
.stDataFrame th,
table th {
    font-weight: 700 !important;
    font-size: 0.73rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: var(--ck-text-muted) !important;
    background: var(--ck-bg-card-alt) !important;
    padding: 10px 12px !important;
    border-bottom: 2px solid var(--ck-border-accent) !important;
}
div[data-testid="stDataFrame"] td,
.stDataFrame td,
table td {
    padding: 8px 12px !important;
    border-bottom: 1px solid var(--ck-border) !important;
    color: var(--ck-text-primary) !important;
}
div[data-testid="stDataFrame"] tr:hover td,
.stDataFrame tr:hover td,
table tr:hover td {
    background: rgba(201,127,0,0.03) !important;
}

/* ── Sidebar refinement ── */
section[data-testid="stSidebar"] {
    font-family: var(--ck-font) !important;
    background: linear-gradient(180deg, var(--ck-bg-sidebar) 0%, var(--ck-bg-sidebar-alt) 100%) !important;
    color: var(--ck-text-on-sidebar) !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stCaption {
    font-size: 0.85rem !important;
    color: color-mix(in srgb, var(--ck-text-on-sidebar) 78%, transparent) !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
    color: var(--ck-text-on-sidebar) !important;
}

/* ── Page-level title refinement ── */
h1 {
    font-family: var(--ck-font) !important;
    letter-spacing: -0.03em !important;
}

/* ── Scrollbar styling ── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: var(--ck-border-accent);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--ck-text-muted);
}

/* ── Animations ── */
@keyframes ck-fade-in {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
.ck-kpi-strip,
.ck-alert,
.ck-hero-section {
    animation: ck-fade-in 0.4s var(--ck-ease) both;
}
.ck-alert { animation-delay: 0.1s; }

</style>
"""


def inject_css() -> None:
    """Inject the Corkysoft design-system CSS into the Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def tier_separator() -> None:
    """Render a subtle gradient separator between screen tiers."""
    st.markdown('<hr class="ck-tier-sep"/>', unsafe_allow_html=True)


def section_title(title: str, subtitle: str | None = None) -> None:
    """Render a styled section title with optional subtitle."""
    st.markdown(f'<div class="ck-section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<div class="ck-section-subtitle">{subtitle}</div>',
            unsafe_allow_html=True,
        )


def hero_section(title: str, subtitle: str | None = None) -> None:
    """Render a complete hero section with title and optional subtitle.

    This must be a single st.markdown call because Streamlit does not allow
    HTML div tags to span across multiple st.markdown invocations.
    """
    from html import escape
    safe_title = escape(title)
    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<div class="ck-section-subtitle">{escape(subtitle)}</div>'
    html = (
        '<div class="ck-hero-section">'
        f'<div class="ck-section-title">{safe_title}</div>'
        f'{subtitle_html}'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def card_open() -> str:
    """Return the opening HTML for a card container."""
    return '<div class="ck-card">'


def card_close() -> str:
    """Return the closing HTML for a card container."""
    return '</div>'


def action_bar_html(label: str = "Actions") -> str:
    """Return the opening HTML for the action bar."""
    return f'<div class="ck-action-bar"><span class="ck-action-label">{label}</span>'


def action_bar_close_html() -> str:
    """Return the closing HTML for the action bar."""
    return '</div>'


def provenance_chip(text: str, icon: str = "🔗") -> None:
    """Render a compact provenance notice as a styled chip."""
    from html import escape
    safe_text = escape(text)
    html = (
        f'<div class="ck-provenance">'
        f'<span class="ck-provenance-icon">{icon}</span>'
        f'{safe_text}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
