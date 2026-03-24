# Planner / Diary Patterns For SB / ITIR

This note extracts reusable workflow patterns from Corkysoft without implying
that SB should become the live removals operations UI.

## Reuse Rule

Reuse the workflow pattern where it is generic.
Do not copy the domain-specific UI or data model where it belongs only to
removals operations.

## Pattern 1: Day / Week Timebox As The Review Frame

- Operators naturally review work as "today" and "this week".
- SB can render timeboxed state summaries that answer "what mattered today" and
  "what remains unresolved this week".

## Pattern 2: Summary -> Drill-Through -> Resolution

- Corkysoft starts from a board, then drills into job, vehicle, staff, or
  invoicing context without losing the original frame.
- SB can expose timeline card -> evidence cluster -> reviewed state drill-down
  rather than flat event dumps.

## Pattern 3: Required vs Utilized

- Operational and financial review both depend on comparing plan against
  actual.
- SB can render "expected vs observed" and "planned vs reviewed" as a first-
  class lens across many domains.

## Pattern 4: Explicit Exception Categories

- Operators need named exception classes, not vague alert lists.
- SB should prefer typed exception families and operator-actionable categories
  over undifferentiated anomaly noise.

## Pattern 5: Persisted Follow-Up Tasks Separate From Core Planning Objects

- Not every follow-up belongs inside `job_segments`.
- SB/ITIR views can benefit from explicit unresolved-follow-up objects rather
  than trying to infer "what still matters" from primary records alone.

## Pattern 6: Visible Uncertainty

- Corkysoft avoids pretending labor and supplier actuals are complete.
- SB should keep confidence, authority class, and provenance visible in review
  surfaces.

## Pattern 7: Finance Follow-Through Tied To Operational Truth

- Invoicing and subcontractor bills become understandable only when tied back
  to actual job/usage context.
- SB should preserve downstream financial review as context-linked state rather
  than detached reporting summaries.

## Pattern 8: Review-State Objects Beat Implicit Completion

- Explicit invoice and bill review objects make the operator's stance visible.
- SB/ITIR should prefer explicit reviewed-state objects when summarizing
  operator decisions and exceptions.

## Good Reuse Targets

- daily and weekly compiled summary lenses
- unresolved exception panels
- required-vs-observed comparison views
- reviewed-state cards with provenance
- follow-up queues tied to stronger source truth

## Bad Reuse Targets

- Planner as a job-dispatch UI
- diary task editing inside SB
- invoice or subcontractor bill state mutation from SB
- any surface that would make SB the authoritative workflow owner
