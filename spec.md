# Corkysoft Remediation Spec

## Objective

Bring Corkysoft's docs, roadmap, Kent AMS workflow, and operator-facing implementation back into sync so planning is reliable and the current product surface is safe to extend.

## Success Criteria

- README, ROADMAP, and Kent docs describe the same current state.
- Operator user stories and lifecycle flows are documented.
- Kent governance, override policy, and multi-truck policy are decision-complete for v1.
- Kent prioritization, hard-block handling, and `dry_run` behavior are corrected in code.
- Kent operator UX is separated from admin/config UX.
- Tests cover the corrected Kent behavior and pass in the project `venv`.

## Non-Goals

- Full production Kent web adapter implementation.
- Full multi-truck solver implementation.
- Broad redesign of unrelated dashboard tabs.

## Current Risks

- Status docs distort prioritization.
- Kent docs mix implemented and planned states.
- Operator workflows are under-specified.
- Kent ranking and hard-block behavior are brittle.
