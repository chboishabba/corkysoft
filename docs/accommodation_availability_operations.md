# Accommodation Availability Operations

This document defines the operational planning direction for using external
accommodation availability signals to support job execution during peak periods.

Use this document as the canonical planning note before implementation.

## Purpose

Some work becomes more operationally attractive or even feasible when nearby
accommodation is available early enough for crews, staging, or overnight
positioning.

Corkysoft should be able to surface availability and booking-pressure context so
operations can:

- plan earlier
- direct crews toward lanes with workable overnight support
- reduce late scramble during peak periods

## Core Idea

Accommodation availability is an operational support signal, not the primary job
planner.

It should help answer:

- are there viable nearby stays for this job / corridor / date window?
- is local availability tightening?
- should operations secure accommodation earlier?
- does this lane become harder to service as availability drops?

## Scope

This direction applies to:

- regional and remote work
- overnight or multi-day routes
- peak-season surge periods
- crew positioning and recovery planning

It does not replace:

- route profitability
- truck / worker assignment
- inventory readiness

It complements them.

## Product Role

Accommodation/availability should surface in:

- `Planner` as contextual pre-commit support
- `Dispatch` as an execution-risk signal
- potentially a future ops coordination surface for bookings and travel logistics

## External Source Direction

External availability providers may include booking-style marketplaces or direct
supplier sources.

Current note:

- local chat-archive fetch was checked for prior `booking.com` discussion
- no direct canonical booking.com thread was resolved from the local archive
- this should be treated as a new product direction, not as a previously locked
  spec

## Planning Signals

The first useful availability signals are:

- nearby accommodation count
- distance from site / route corridor
- date-window availability
- price band
- cancellation flexibility
- warning when availability is low or falling early

## Operational Uses

### Planner

- show accommodation pressure for likely overnight jobs
- surface routes where early booking materially improves feasibility
- treat availability as contextual support, not as the sole ranking driver

### Dispatch

- flag jobs where crew travel/accommodation is becoming risky
- show whether accommodation has been arranged or still needs action

### Management

- identify corridors or seasons where accommodation scarcity affects serviceability
- decide when to pre-book during peak periods

## Recommended Implementation Order

1. document the required fields and workflow
2. ingest a lightweight availability summary from one external source
3. surface availability risk in Planner and Dispatch
4. add explicit booking-tracking or handoff workflow if it proves operationally useful

## Notes

- Earlier booking is likely the main value, not consumer-style search UX.
- The feature should drive operational foresight, not turn Corkysoft into a travel portal.
