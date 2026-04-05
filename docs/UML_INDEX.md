# UML Index

The UML suite is generated from the repository's internal Python import graph.

## Entrypoint

- [supermega_01.puml](rendered/plantuml/supermega_01.puml) | [supermega_01.svg](rendered/svg/supermega_01.svg)

## Child Views

- [entrypoints_integrations.puml](rendered/plantuml/entrypoints_integrations.puml) | [entrypoints_integrations.svg](rendered/svg/entrypoints_integrations.svg): Entrypoints (4 modules)
- [dashboard_shell.puml](rendered/plantuml/dashboard_shell.puml) | [dashboard_shell.svg](rendered/svg/dashboard_shell.svg): Dashboard Shell (11 modules)
- [workflow_surfaces.puml](rendered/plantuml/workflow_surfaces.puml) | [workflow_surfaces.svg](rendered/svg/workflow_surfaces.svg): Workflow Surfaces (29 modules)
- [analytics_persistence.puml](rendered/plantuml/analytics_persistence.puml) | [analytics_persistence.svg](rendered/svg/analytics_persistence.svg): Analytics + Persistence (49 modules)
- [quote_routing_core.puml](rendered/plantuml/quote_routing_core.puml) | [quote_routing_core.svg](rendered/svg/quote_routing_core.svg): Quote + Routing Core (7 modules)
- [integrations_extensions.puml](rendered/plantuml/integrations_extensions.puml) | [integrations_extensions.svg](rendered/svg/integrations_extensions.svg): Integrations + Extensions (27 modules)

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
python scripts/build_supermega_uml.py --render
python scripts/build_supermega_uml.py --check --render
```

The rendered SVG artifacts are generated with PlantUML when available.


## Builder

- `python scripts/build_supermega_uml.py` regenerates the PlantUML suite.
- `python scripts/build_supermega_uml.py --check` validates that generated artifacts are up to date.
- `python scripts/build_supermega_uml.py --render` additionally renders SVG artifacts.
- `python scripts/build_supermega_uml.py --check --render` also validates rendered SVG freshness.
