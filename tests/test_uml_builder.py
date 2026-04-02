from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _load_builder_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_supermega_uml.py"
    spec = importlib.util.spec_from_file_location("build_supermega_uml", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_artifacts_includes_supermega_and_shell() -> None:
    builder = _load_builder_module()

    artifacts = builder.generate_artifacts()

    assert Path("docs/rendered/plantuml/supermega_01.puml") in {
        path.relative_to(builder.REPO_ROOT) for path in artifacts
    }
    assert Path("docs/architecture_dashboard_shell.puml") in {
        path.relative_to(builder.REPO_ROOT) for path in artifacts
    }


def test_builder_assigns_all_discovered_modules() -> None:
    builder = _load_builder_module()

    graph = builder.build_import_graph(builder.REPO_ROOT)

    assert builder.unassigned_modules(graph) == []


def test_supermega_contains_real_cross_domain_links() -> None:
    builder = _load_builder_module()

    artifacts = builder.generate_artifacts()
    supermega = artifacts[builder.REPO_ROOT / "docs" / "rendered" / "plantuml" / "supermega_01.puml"]

    assert "dashboard_shell --> workflow_surfaces" in supermega
    assert (
        "workflow_surfaces --> analytics_persistence : "
        "quote_builder -> price_distribution"
    ) in supermega
    assert "route_maps -> price_distribution" in supermega
    assert (
        "workflow_surfaces --> quote_routing_core : "
        "quote_builder -> quote_service"
    ) in supermega
    assert "workflow_surfaces --> integrations_extensions : calls -> call_ops" in supermega
    assert "quote_routing_core --> analytics_persistence : quote_service -> kent_ams_import" in supermega


def test_child_diagrams_have_plantuml_wrappers() -> None:
    builder = _load_builder_module()

    artifacts = builder.generate_artifacts()
    child_paths = [
        builder.REPO_ROOT / "docs" / "rendered" / "plantuml" / spec.output_name
        for spec in builder.DOMAIN_SPECS
    ]

    for path in child_paths:
        content = artifacts[path]
        assert content.startswith("@startuml\n")
        assert content.endswith("@enduml\n")


def test_child_diagrams_anchor_external_links_to_real_source_modules() -> None:
    builder = _load_builder_module()

    artifacts = builder.generate_artifacts()
    workflow_surfaces = artifacts[
        builder.REPO_ROOT / "docs" / "rendered" / "plantuml" / "workflow_surfaces.puml"
    ]

    assert (
        "dashboard_components_quote_builder ..> external_analytics_persistence : "
        "quote_builder -> price_distribution"
    ) in workflow_surfaces
    assert (
        "dashboard_components_route_maps ..> external_analytics_persistence : "
        "route_maps -> price_distribution"
    ) in workflow_surfaces
    assert (
        "dashboard_components_inventory ..> external_analytics_persistence : "
        "inventory -> db"
    ) in workflow_surfaces
    assert (
        "dashboard_components_calls ..> external_integrations_extensions : "
        "calls -> call_ops"
    ) in workflow_surfaces


def test_index_includes_rendered_svg_entrypoints() -> None:
    builder = _load_builder_module()
    index_path = builder.REPO_ROOT / "docs" / "UML_INDEX.md"
    artifacts = builder.generate_artifacts()

    assert index_path in artifacts
    index = artifacts[index_path]
    assert "[supermega_01.svg](rendered/svg/supermega_01.svg)" in index
    assert "[dashboard_shell.svg](rendered/svg/dashboard_shell.svg)" in index
    assert "`python scripts/build_supermega_uml.py --render`" in index


def test_renderable_plantuml_paths_and_svg_targets() -> None:
    builder = _load_builder_module()
    artifacts = builder.generate_artifacts()
    renderables = builder.renderable_plantuml_paths(artifacts)

    assert (
        builder.REPO_ROOT / "docs" / "rendered" / "plantuml" / "supermega_01.puml"
    ) in renderables
    assert (
        builder.REPO_ROOT / "docs" / "architecture_dashboard_shell.puml"
    ) in renderables

    for source_path in renderables:
        output_path = builder.rendered_svg_path(source_path)
        assert output_path.parent == builder.REPO_ROOT / "docs" / "rendered" / "svg"
        assert output_path.suffix == ".svg"


def test_render_requires_plantuml_binary() -> None:
    builder = _load_builder_module()

    try:
        builder.render_plantuml_artifacts(
            [builder.REPO_ROOT / "docs" / "rendered" / "plantuml" / "supermega_01.puml"],
            command="definitely-not-a-real-plantuml-binary",
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError when PlantUML binary is missing")


def test_render_plantuml_artifacts_requires_binary(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _load_builder_module()
    source = tmp_path / "diagram.puml"
    source.write_text("@startuml\nA -> B\n@enduml\n", encoding="utf-8")

    monkeypatch.setattr(builder.shutil, "which", lambda _: None)

    with pytest.raises(FileNotFoundError):
        builder.render_plantuml_artifacts([source], command="plantuml")
