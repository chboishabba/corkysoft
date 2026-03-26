# Contributor Docs Sync

Use this note when shipping feature or refactor work so the docs stay aligned
with the codebase.

## Minimum update rule

After any meaningful behavior, workflow, or architecture change:

1. update [README.md](../README.md) if the user-facing product story, current
   status, or entry points changed
2. update [ROADMAP.md](../ROADMAP.md) if status, scope, or next-step wording
   changed
3. update [docs/progress_status_board.md](progress_status_board.md) when an item
   moves between pending, in progress, or implemented
4. update any feature-specific spec that is now the canonical truth for that
   surface
5. record the change in [CHANGELOG.md](../CHANGELOG.md)

## When to update architecture docs

Update [docs/modules.md](modules.md) when:
- a large module is split into new files
- an entry surface becomes a facade/composition layer
- a new component or analytics subsystem becomes the obvious ownership boundary

Update [docs/positioning.md](positioning.md) when:
- product differentiation changes
- a new operational/commercial workflow becomes real
- integration strategy or system boundaries change

Update [docs/corkysoft_sb_itir_coverage_audit.md](corkysoft_sb_itir_coverage_audit.md) when:
- Corkysoft takes ownership of a workflow previously marked conceptual
- a downstream export/contract becomes implemented or materially clearer

## Refactor guidance

Do not document every file move as if it were a product feature.

Document refactors when they materially change:
- module ownership
- public import surfaces
- main entry points
- contributor expectations about where new logic belongs

## Practical standard

The goal is not perfect prose. The goal is to avoid these failure modes:
- README describing features that are no longer current
- ROADMAP marking implemented work as future work
- module docs pointing contributors at files that are no longer the right place
- architecture docs missing a newly real boundary between Corkysoft and
  downstream systems
