from dashboard import theme


def test_theme_css_preserves_streamlit_material_icons() -> None:
    css = theme._CSS

    assert ".stMarkdown, .stText, p, label, .stCaption," in css
    assert ".stMarkdown, .stText, p, span, label, .stCaption," not in css
    assert 'span[data-testid="stIconMaterial"]' in css
    assert "Material Symbols Rounded" in css
    assert "Material Symbols Outlined" in css
    assert 'font-feature-settings: "liga" 1' in css
    assert "Material+Symbols+Outlined:opsz,wght,FILL,GRAD@" in css
    assert "Material+Symbols+Rounded:opsz,wght,FILL,GRAD@" in css


def test_theme_css_keeps_light_surface_tokens() -> None:
    css = theme._CSS

    assert "@media (prefers-color-scheme: dark)" not in css
    assert "--ck-bg-primary: Canvas;" in css
    assert "var(--ck-bg-primary-alt)" in css


def test_theme_css_uses_theme_adaptive_system_colours() -> None:
    css = theme._CSS

    assert "color-mix(in srgb, Canvas 90%, CanvasText 10%)" in css
    assert "--ck-text-primary: CanvasText;" in css
    assert "--ck-bg-sidebar-alt:" in css
    assert not hasattr(theme, "_THEME_SYNC_SCRIPT")
