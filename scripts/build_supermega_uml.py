#!/usr/bin/env python3
"""Build PlantUML child diagrams plus an integrated supermega view.

The builder derives links from the repository's internal Python import graph so
the UML suite stays aligned with real module boundaries instead of relying on
manually curated arrows alone.
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
RENDERED_PLANTUML_DIR = DOCS_DIR / "rendered" / "plantuml"
RENDERED_SVG_DIR = DOCS_DIR / "rendered" / "svg"

EXCLUDED_PATH_PARTS = {
    "__pycache__",
    "venv",
    ".venv",
    "node_modules",
    "playwright-report",
    "test-results",
}

ENTRYPOINT_MODULES = (
    "map_jobs",
    "profit_optimizer",
    "quick_quote",
    "routes_to_sqlite",
)


@dataclass(frozen=True)
class DomainSpec:
    key: str
    title: str
    output_name: str
    prefixes: tuple[str, ...]
    summary_modules: tuple[str, ...]


DOMAIN_SPECS: tuple[DomainSpec, ...] = (
    DomainSpec(
        key="entrypoints",
        title="Entrypoints",
        output_name="entrypoints_integrations.puml",
        prefixes=ENTRYPOINT_MODULES,
        summary_modules=(
            "routes_to_sqlite",
            "quick_quote",
            "profit_optimizer",
            "map_jobs",
        ),
    ),
    DomainSpec(
        key="dashboard_shell",
        title="Dashboard Shell",
        output_name="dashboard_shell.puml",
        prefixes=(
            "dashboard.app",
            "dashboard.auth_ui",
            "dashboard.data",
            "dashboard.data_controls",
            "dashboard.layout_state",
            "dashboard.map_provider",
            "dashboard.query_params",
            "dashboard.shell",
            "dashboard.state",
            "dashboard.tab_registry",
            "dashboard.theme",
        ),
        summary_modules=(
            "dashboard.app",
            "dashboard.auth_ui",
            "dashboard.data_controls",
            "dashboard.tab_registry",
        ),
    ),
    DomainSpec(
        key="workflow_surfaces",
        title="Workflow Surfaces",
        output_name="workflow_surfaces.puml",
        prefixes=("dashboard.components", "dashboard.views"),
        summary_modules=(
            "dashboard.views.quote_view",
            "dashboard.views.pricing_intelligence_view",
            "dashboard.views.network_view",
            "dashboard.views.operations_view",
            "dashboard.views.admin_view",
        ),
    ),
    DomainSpec(
        key="analytics_persistence",
        title="Analytics + Persistence",
        output_name="analytics_persistence.puml",
        prefixes=("analytics",),
        summary_modules=(
            "analytics.price_distribution",
            "analytics.route_map_prep",
            "analytics.operations_diary",
            "analytics.db",
        ),
    ),
    DomainSpec(
        key="quote_routing_core",
        title="Quote + Routing Core",
        output_name="quote_routing_core.puml",
        prefixes=(
            "corkysoft.au_address",
            "corkysoft.pricing",
            "corkysoft.quote_service",
            "corkysoft.repo",
            "corkysoft.routing",
            "corkysoft.schema",
        ),
        summary_modules=(
            "corkysoft.quote_service",
            "corkysoft.routing",
            "corkysoft.repo",
            "corkysoft.pricing",
        ),
    ),
    DomainSpec(
        key="integrations_extensions",
        title="Integrations + Extensions",
        output_name="integrations_extensions.puml",
        prefixes=(
            "corkysoft.api",
            "corkysoft.api_calls",
            "corkysoft.api_kent",
            "corkysoft.api_labor",
            "corkysoft.api_operations",
            "corkysoft.api_shared",
            "corkysoft.call_ops",
            "corkysoft.call_ops_actions",
            "corkysoft.call_ops_core",
            "corkysoft.call_ops_transcripts",
            "corkysoft.call_ops_worker_time",
            "corkysoft.cost_model",
            "corkysoft.importers",
            "corkysoft.mcp",
            "corkysoft.whisperx_adapter",
            "corkysoft.src.dashboard",
        ),
        summary_modules=(
            "corkysoft.api",
            "corkysoft.call_ops",
            "corkysoft.mcp",
            "corkysoft.importers.jobs_api",
        ),
    ),
)
DOMAIN_SPEC_BY_KEY = {spec.key: spec for spec in DOMAIN_SPECS}


def _plantuml_safe_alias(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value)


def _discover_python_modules(repo_root: Path) -> dict[str, Path]:
    modules: dict[str, Path] = {}
    package_roots = ("dashboard", "analytics", "corkysoft")
    for package_root in package_roots:
        for path in sorted((repo_root / package_root).rglob("*.py")):
            if any(part in EXCLUDED_PATH_PARTS for part in path.parts):
                continue
            relative = path.relative_to(repo_root)
            parts = relative.with_suffix("").parts
            if parts[-1] == "__init__" and len(parts) == 2:
                continue
            module_name = ".".join(parts[:-1]) if parts[-1] == "__init__" else ".".join(parts)
            modules[module_name] = path

    for module_name in ENTRYPOINT_MODULES:
        path = repo_root / f"{module_name}.py"
        if path.exists():
            modules[module_name] = path

    return modules


def _match_known_module(name: str, known_modules: Iterable[str]) -> str | None:
    matches = [
        module
        for module in known_modules
        if name == module or name.startswith(f"{module}.")
    ]
    if not matches:
        return None
    return max(matches, key=len)


def _resolve_import_from(module_name: str, node: ast.ImportFrom) -> str:
    package_parts = module_name.split(".")[:-1]
    if node.level == 0:
        return node.module or ""
    ascend = max(len(package_parts) - node.level + 1, 0)
    base_parts = package_parts[:ascend]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


def build_import_graph(repo_root: Path = REPO_ROOT) -> dict[str, set[str]]:
    """Return the internal import graph keyed by module name."""

    modules = _discover_python_modules(repo_root)
    known_modules = tuple(sorted(modules))
    graph: dict[str, set[str]] = {module: set() for module in modules}

    for module_name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    match = _match_known_module(alias.name, known_modules)
                    if match and match != module_name:
                        imports.add(match)
            elif isinstance(node, ast.ImportFrom):
                resolved = _resolve_import_from(module_name, node)
                match = _match_known_module(resolved, known_modules)
                if match and match != module_name:
                    imports.add(match)
        graph[module_name] = imports

    return graph


def _module_domain(module_name: str) -> str:
    for spec in DOMAIN_SPECS:
        if any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in spec.prefixes
        ):
            return spec.key
    raise KeyError(f"Module {module_name!r} is not assigned to a UML domain")


def unassigned_modules(graph: Mapping[str, set[str]]) -> list[str]:
    return sorted(module for module in graph if not _can_assign_domain(module))


def _can_assign_domain(module_name: str) -> bool:
    try:
        _module_domain(module_name)
    except KeyError:
        return False
    return True


def build_domain_module_map(graph: Mapping[str, set[str]]) -> dict[str, list[str]]:
    domain_map = {spec.key: [] for spec in DOMAIN_SPECS}
    for module_name in sorted(graph):
        domain_map[_module_domain(module_name)].append(module_name)
    return domain_map


def _module_label(module_name: str) -> str:
    if "." not in module_name:
        return f"{module_name}.py"
    return module_name


def _external_domain_alias(domain_key: str) -> str:
    return f"external_{_plantuml_safe_alias(domain_key)}"


def _module_edges_for_domain_pair(
    graph: Mapping[str, set[str]],
    source_domain: str,
    target_domain: str,
) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for source_module, targets in graph.items():
        if _module_domain(source_module) != source_domain:
            continue
        for target_module in sorted(targets):
            if _module_domain(target_module) == target_domain:
                edges.append((source_module, target_module))
    source_spec = DOMAIN_SPEC_BY_KEY[source_domain]
    target_spec = DOMAIN_SPEC_BY_KEY[target_domain]
    return sorted(
        edges,
        key=lambda edge: (
            _summary_priority(edge[0], source_spec),
            _summary_priority(edge[1], target_spec),
            edge[0],
            edge[1],
        ),
    )


def _summary_priority(module_name: str, spec: DomainSpec) -> tuple[int, str]:
    for index, summary_module in enumerate(spec.summary_modules):
        if module_name == summary_module or module_name.startswith(f"{summary_module}."):
            return (index, module_name)
    return (len(spec.summary_modules), module_name)


def _representative_edge_labels(
    graph: Mapping[str, set[str]],
    source_domain: str,
    target_domain: str,
) -> str:
    limit = 5 if (source_domain, target_domain) == ("dashboard_shell", "workflow_surfaces") else 3
    edges = _module_edges_for_domain_pair(graph, source_domain, target_domain)
    if not edges:
        return "imports"

    # Prefer one example per source module first so supermega links show
    # multiple child feeders instead of repeating one dominant child module.
    selected: list[tuple[str, str]] = []
    seen_sources: set[str] = set()
    for source, target in edges:
        if source in seen_sources:
            continue
        selected.append((source, target))
        seen_sources.add(source)
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for source, target in edges:
            if (source, target) in selected:
                continue
            selected.append((source, target))
            if len(selected) >= limit:
                break

    labels = [f"{source.split('.')[-1]} -> {target.split('.')[-1]}" for source, target in selected]
    return "\\n".join(labels)


def _domain_summary_label(spec: DomainSpec) -> str:
    lines = [spec.title]
    for module_name in spec.summary_modules:
        lines.append(
            module_name.replace("dashboard.components.", "components.").replace(
                "dashboard.views.", "views."
            )
        )
    return "\\n".join(lines)


def render_child_diagram(
    spec: DomainSpec,
    graph: Mapping[str, set[str]],
    domain_modules: Mapping[str, Sequence[str]],
) -> str:
    modules = list(domain_modules[spec.key])
    aliases = {module: _plantuml_safe_alias(module) for module in modules}
    lines = [
        "@startuml",
        f"title Corkysoft {spec.title}",
        "left to right direction",
        "skinparam componentStyle rectangle",
        "skinparam shadowing false",
        "skinparam wrapWidth 220",
        "skinparam maxMessageSize 180",
        f'package "{spec.title}" {{',
    ]
    for module in modules:
        lines.append(f'  [{_module_label(module)}] as {aliases[module]}')
    lines.append("}")

    for source in modules:
        for target in sorted(graph[source]):
            if _module_domain(target) == spec.key:
                lines.append(f"{aliases[source]} --> {aliases[target]}")

    external_domains = sorted(
        {
            _module_domain(target)
            for module in modules
            for target in graph[module]
            if _module_domain(target) != spec.key
        }
    )
    for domain_key in external_domains:
        target_spec = DOMAIN_SPEC_BY_KEY[domain_key]
        external_alias = _external_domain_alias(domain_key)
        lines.append(f'[{target_spec.title}] as {external_alias}')
        domain_edges = _module_edges_for_domain_pair(graph, spec.key, domain_key)
        emitted_sources: set[str] = set()
        for source_module, _target_module in domain_edges:
            if source_module in emitted_sources:
                continue
            source_specific_edges = [
                (edge_source, edge_target)
                for edge_source, edge_target in domain_edges
                if edge_source == source_module
            ]
            source_labels = [
                f"{edge_source.split('.')[-1]} -> {edge_target.split('.')[-1]}"
                for edge_source, edge_target in source_specific_edges[:3]
            ]
            source_label_text = "\\n".join(source_labels)
            lines.append(
                f"{aliases[source_module]} ..> {external_alias} : {source_label_text}"
            )
            emitted_sources.add(source_module)

    lines.append("@enduml")
    return "\n".join(lines) + "\n"


def render_supermega(
    graph: Mapping[str, set[str]],
    domain_modules: Mapping[str, Sequence[str]],
) -> str:
    lines = [
        "@startuml",
        "title Corkysoft Integrated Supermega Map",
        "left to right direction",
        "skinparam componentStyle uml2",
        "skinparam shadowing false",
        "skinparam wrapWidth 240",
        "skinparam maxMessageSize 180",
        'actor "Operator / Planner / Admin" as Operator',
    ]

    for spec in DOMAIN_SPECS:
        alias = _plantuml_safe_alias(spec.key)
        lines.append(f'package "{spec.title}" {{')
        lines.append(f'  [{_domain_summary_label(spec)}] as {alias}')
        lines.append("}")

    lines.append("Operator --> dashboard_shell")

    for source_spec in DOMAIN_SPECS:
        for target_spec in DOMAIN_SPECS:
            if source_spec.key == target_spec.key:
                continue
            edges = _module_edges_for_domain_pair(graph, source_spec.key, target_spec.key)
            if not edges:
                continue
            source_alias = _plantuml_safe_alias(source_spec.key)
            target_alias = _plantuml_safe_alias(target_spec.key)
            label = _representative_edge_labels(graph, source_spec.key, target_spec.key)
            lines.append(f"{source_alias} --> {target_alias} : {label}")

    lines.extend(
        [
            "note bottom of dashboard_shell",
            "Shell owns composition, auth gate, sidebar controls,",
            "query-param hydration, and stable tab selection.",
            "end note",
            "note bottom of workflow_surfaces",
            "Five shell views live in dashboard/views/*,",
            "with component leaf surfaces under dashboard/components/*.",
            "end note",
            "note bottom of analytics_persistence",
            "Analytics owns prep, diary logic, map shaping,",
            "and database-facing operational truth helpers.",
            "end note",
            "note bottom of quote_routing_core",
            "Quote calculation and routing remain producer-owned",
            "inside corkysoft rather than inside Streamlit surfaces.",
            "end note",
        ]
    )

    lines.append("@enduml")
    return "\n".join(lines) + "\n"


def render_index(domain_modules: Mapping[str, Sequence[str]]) -> str:
    lines = [
        "# UML Index",
        "",
        "The UML suite is generated from the repository's internal Python import graph.",
        "",
        "## Entrypoint",
        "",
        "- [supermega_01.puml](rendered/plantuml/supermega_01.puml) | "
        "[supermega_01.svg](rendered/svg/supermega_01.svg)",
        "",
        "## Child Views",
        "",
    ]
    for spec in DOMAIN_SPECS:
        lines.append(
            f"- [{spec.output_name}](rendered/plantuml/{spec.output_name}) | "
            f"[{Path(spec.output_name).with_suffix('.svg')}]"
            f"(rendered/svg/{Path(spec.output_name).with_suffix('.svg')}): "
            f"{spec.title} ({len(domain_modules[spec.key])} modules)"
        )
    lines.extend(
        [
            "",
            "## Rendered Assets",
            "- [supermega_01.svg](rendered/svg/supermega_01.svg)",
            "- [entrypoints_integrations.svg](rendered/svg/entrypoints_integrations.svg)",
            "- [dashboard_shell.svg](rendered/svg/dashboard_shell.svg)",
            "- [workflow_surfaces.svg](rendered/svg/workflow_surfaces.svg)",
            "- [analytics_persistence.svg](rendered/svg/analytics_persistence.svg)",
            "- [quote_routing_core.svg](rendered/svg/quote_routing_core.svg)",
            "- [integrations_extensions.svg](rendered/svg/integrations_extensions.svg)",
            "- [architecture_dashboard_shell.svg](rendered/svg/architecture_dashboard_shell.svg)",
            "",
            "```",
            "python scripts/build_supermega_uml.py --render",
            "python scripts/build_supermega_uml.py --check --render",
            "```",
            "",
            "The rendered SVG artifacts are generated with PlantUML when available.",
            "",
        ]
    )
    lines.extend(
        [
        "",
        "## Builder",
        "",
        "- `python scripts/build_supermega_uml.py` regenerates the PlantUML suite.",
        "- `python scripts/build_supermega_uml.py --check` validates that generated artifacts are up to date.",
        "- `python scripts/build_supermega_uml.py --render` additionally renders SVG artifacts.",
        "- `python scripts/build_supermega_uml.py --check --render` also validates rendered SVG freshness.",
            "",
        ]
    )
    return "\n".join(lines)


def renderable_plantuml_paths(artifacts: Mapping[Path, str]) -> list[Path]:
    paths = [
        path
        for path in artifacts
        if path.suffix == ".puml"
        and (
            path.parent == RENDERED_PLANTUML_DIR
            or path == DOCS_DIR / "architecture_dashboard_shell.puml"
        )
    ]
    return sorted(paths)


def rendered_svg_path(source_path: Path) -> Path:
    if source_path == DOCS_DIR / "architecture_dashboard_shell.puml":
        return RENDERED_SVG_DIR / "architecture_dashboard_shell.svg"
    if source_path.parent == RENDERED_PLANTUML_DIR:
        return RENDERED_SVG_DIR / source_path.with_suffix(".svg").name
    raise ValueError(f"Unsupported UML source for rendering: {source_path}")


def rendered_artifacts_out_of_date(artifacts: Mapping[Path, str]) -> list[Path]:
    stale: list[Path] = []
    for source_path in renderable_plantuml_paths(artifacts):
        output_path = rendered_svg_path(source_path)
        if not output_path.exists():
            stale.append(output_path)
            continue
        if output_path.stat().st_mtime < source_path.stat().st_mtime:
            stale.append(output_path)
    return stale


def render_plantuml_artifacts(
    plantuml_paths: Sequence[Path],
    *,
    format: str = "svg",
    command: str = "plantuml",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[Path]:
    if not plantuml_paths:
        return []
    if not shutil.which(command):
        raise FileNotFoundError(
            "plantuml not found. Install PlantUML and Java (if required) to render diagrams."
        )

    RENDERED_SVG_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for source_path in sorted(plantuml_paths):
        output_path = rendered_svg_path(source_path)
        cwd = source_path.parent
        output_dir = "../svg"
        if source_path == DOCS_DIR / "architecture_dashboard_shell.puml":
            output_dir = "rendered/svg"
        env = os.environ.copy()
        env["DISPLAY"] = ""
        result = runner(
            [command, f"-t{format}", "-o", output_dir, source_path.name],
            cwd=str(cwd),
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"PlantUML rendering failed for {source_path.relative_to(REPO_ROOT)}: "
                f"{result.stdout.rstrip()} {result.stderr.rstrip()}".strip()
            )
        outputs.append(output_path)
    return outputs


def generate_artifacts(repo_root: Path = REPO_ROOT) -> dict[Path, str]:
    graph = build_import_graph(repo_root)
    unassigned = unassigned_modules(graph)
    if unassigned:
        names = ", ".join(unassigned)
        raise ValueError(f"UML builder has unassigned modules: {names}")

    domain_modules = build_domain_module_map(graph)
    artifacts: dict[Path, str] = {}

    for spec in DOMAIN_SPECS:
        child = render_child_diagram(spec, graph, domain_modules)
        output_path = repo_root / "docs" / "rendered" / "plantuml" / spec.output_name
        artifacts[output_path] = child
        if spec.key == "dashboard_shell":
            artifacts[repo_root / "docs" / "architecture_dashboard_shell.puml"] = child

    artifacts[repo_root / "docs" / "rendered" / "plantuml" / "supermega_01.puml"] = render_supermega(
        graph,
        domain_modules,
    )
    artifacts[repo_root / "docs" / "UML_INDEX.md"] = render_index(domain_modules)
    return artifacts


def write_artifacts(artifacts: Mapping[Path, str]) -> None:
    for path, content in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def artifacts_out_of_date(artifacts: Mapping[Path, str]) -> list[Path]:
    stale: list[Path] = []
    for path, expected in artifacts.items():
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            stale.append(path)
    return stale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated UML artifacts differ from the current files.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render PlantUML diagrams after artifact generation.",
    )
    parser.add_argument(
        "--render-format",
        default="svg",
        help="PlantUML output format when rendering (default: svg).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = generate_artifacts()
    if args.check:
        stale = artifacts_out_of_date(artifacts)
        if args.render:
            stale.extend(rendered_artifacts_out_of_date(artifacts))
        if stale:
            for path in stale:
                print(f"out of date: {path.relative_to(REPO_ROOT)}")
            return 1
        print("UML artifacts are up to date.")
        if args.render:
            print("Rendered UML assets are up to date.")
        return 0

    write_artifacts(artifacts)
    if args.render:
        try:
            render_plantuml_artifacts(
                renderable_plantuml_paths(artifacts),
                format=args.render_format,
            )
            print("Rendered UML assets to", RENDERED_SVG_DIR.relative_to(REPO_ROOT))
        except FileNotFoundError as exc:
            print(str(exc))
            return 1
        except RuntimeError as exc:
            print(str(exc))
            return 1
    for path in artifacts:
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
