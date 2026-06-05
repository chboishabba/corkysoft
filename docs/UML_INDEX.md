# UML Index

The generated UML suite is generated from the repository's internal Python
import graph. Hand-authored service-flow diagrams are listed separately below
because they describe actor/story paths rather than module imports.

Freshness note:

- Last governance review recorded: 2026-06-04.
- Authoritative check command: `venv/bin/python scripts/build_supermega_uml.py --check --render`.
- This suite describes the repository import/control structure. It is not a
  replacement for the still-needed truck/server/cloud deployment architecture
  diagram tracked in `ROADMAP.md`.

## Entrypoint

- [supermega_01.puml](rendered/plantuml/supermega_01.puml) | [supermega_01.svg](rendered/svg/supermega_01.svg)

## Child Views

- [entrypoints_integrations.puml](rendered/plantuml/entrypoints_integrations.puml) | [entrypoints_integrations.svg](rendered/svg/entrypoints_integrations.svg): Entrypoints (4 modules)
- [dashboard_shell.puml](rendered/plantuml/dashboard_shell.puml) | [dashboard_shell.svg](rendered/svg/dashboard_shell.svg): Dashboard Shell (13 modules)
- [workflow_surfaces.puml](rendered/plantuml/workflow_surfaces.puml) | [workflow_surfaces.svg](rendered/svg/workflow_surfaces.svg): Workflow Surfaces (29 modules)
- [analytics_persistence.puml](rendered/plantuml/analytics_persistence.puml) | [analytics_persistence.svg](rendered/svg/analytics_persistence.svg): Analytics + Persistence (49 modules)
- [quote_routing_core.puml](rendered/plantuml/quote_routing_core.puml) | [quote_routing_core.svg](rendered/svg/quote_routing_core.svg): Quote + Routing Core (7 modules)
- [integrations_extensions.puml](rendered/plantuml/integrations_extensions.puml) | [integrations_extensions.svg](rendered/svg/integrations_extensions.svg): Integrations + Extensions (27 modules)

## Service Blueprint / Story Path Views

- [service_blueprint_flows.puml](diagrams/service_blueprint_flows.puml): Hand-authored
  actor, shell, interaction, and authority-gate diagram keyed to the story path
  IDs in [Service Blueprint](service_blueprint.md#story-path-index).
- [service_blueprint_flows.svg](diagrams/service_blueprint_flows.svg): Rendered vector view.
- [service_blueprint_flows.png](diagrams/service_blueprint_flows.png): Rendered raster view.

## Rendered Assets
- [supermega_01.svg](rendered/svg/supermega_01.svg)
- [entrypoints_integrations.svg](rendered/svg/entrypoints_integrations.svg)
- [dashboard_shell.svg](rendered/svg/dashboard_shell.svg)
- [workflow_surfaces.svg](rendered/svg/workflow_surfaces.svg)
- [analytics_persistence.svg](rendered/svg/analytics_persistence.svg)
- [quote_routing_core.svg](rendered/svg/quote_routing_core.svg)
- [integrations_extensions.svg](rendered/svg/integrations_extensions.svg)
- [architecture_dashboard_shell.svg](rendered/svg/architecture_dashboard_shell.svg)

```
venv/bin/python scripts/build_supermega_uml.py --render
venv/bin/python scripts/build_supermega_uml.py --check --render
```

The rendered SVG artifacts are generated with PlantUML when available.


## Builder

- `venv/bin/python scripts/build_supermega_uml.py` regenerates the PlantUML suite.
- `venv/bin/python scripts/build_supermega_uml.py --check` validates that generated artifacts are up to date.
- `venv/bin/python scripts/build_supermega_uml.py --render` additionally renders SVG artifacts.
- `venv/bin/python scripts/build_supermega_uml.py --check --render` also validates rendered SVG freshness.
