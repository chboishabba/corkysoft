# Live Network Overview

This document describes the **current MVP intent** for the Live Network view
and separates that from future enhancements.

## Current MVP

Purpose:
- give operations and commercial teams a quick visual read of live trucks,
  active routes, and lane profitability context

Current expected behavior:
- display active trucks and routes from `truck_positions` and `active_routes`
- overlay profitability context where the underlying lane/job data exists
- support dashboard-level filters already exposed elsewhere in the app
- remain usable even when some geometry or profitability data is missing

Current non-goals:
- guaranteed auto-refresh or polling cadence
- lane-click side panels
- advanced drill-down workflows
- real-time SLA alerting beyond what current datasets support

## Current Data Dependencies

- `truck_positions`
- `active_routes`
- `jobs`
- `historical_jobs`
- derived profitability/lane data where available

If profitability data is absent:
- the view should still function as a live network map
- profitability coloring/drill-down should degrade gracefully

## Operator Story

Primary user:
- dispatcher or operations manager scanning live state

Primary questions:
- where are active trucks right now?
- which active corridors look commercially strong or weak?
- do any current moves line up with work we should prioritize next?

Expected outcome:
- quick situational awareness, not deep investigation

## Future Enhancements

These are not guaranteed by the current implementation:

- timed auto-refresh/polling behavior
- click-through side panels
- corridor-level sparklines and recent exceptions
- dedicated export flows from the live view
- richer SLA/incident overlays
- state/national closure, traffic, and weather overlays normalized into a
  common situational-awareness layer

If any of these are implemented, this document should be updated from
"future enhancement" to "current MVP" only after code and tests exist.
