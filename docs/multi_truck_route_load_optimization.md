# Multi-Truck Route and Load Optimization

This document defines the operational optimization problem for combining:

- multi-job pickup/drop sequencing,
- per-truck capacity constraints,
- optional cross-dock / truck-to-truck transfer operations,
- route cost and margin optimization,
- peak-season throughput priorities.

It is a design and planning spec. It does **not** imply full implementation is complete.

## Problem Statement

Corkysoft needs to optimize plans like:

1. Truck A loads job 1 at origin `i`.
2. Truck A continues to origin `ii` and loads job 2.
3. Deliver job 2 to destination `ii`.
4. Deliver job 1 to destination `i`.

Generalized form:

- many jobs can share one truck,
- one job can require multiple trucks,
- loads can be transferred between trucks at approved transfer points,
- priorities vary by profitability, deadlines, and peak-season capacity pressure.

## Why The "Three Stacks" Analogy Is Useful

The Tower-of-Hanoi analogy is directionally useful for understanding constrained
reordering, but this problem is harder because it also includes:

- spatial routing costs (distance/time),
- time windows and due dates,
- heterogeneous truck capacities,
- partially divisible loads,
- optional transfer points,
- labor constraints.

So this should be treated as a hybrid:

- vehicle routing problem (VRP),
- pickup and delivery with time windows (PDPTW),
- load/bin-packing problem,
- dynamic dispatch under uncertainty.

## Optimization Objectives

Primary objectives (ranked):

1. Maximize realized margin/profit.
2. Preserve service reliability (deadline compliance, low disruption).
3. Maximize peak-season throughput (including controlled over-capacity operation when approved).
4. Minimize deadhead distance and unnecessary transfers.

Secondary objectives:

- keep plans explainable for operators,
- reduce re-handling risk and claims exposure,
- improve reuse of en-route spare capacity.

## Core Constraints

- Truck volume/weight capacity (`capacity_m3`, load limits).
- Worker/crew availability and qualifications.
- Pickup/drop precedence (`pickup` before `delivery`).
- Time windows (`due_at`, customer windows, depot cutoffs).
- Allowed transfer locations and handling limits.
- Route feasibility from provider travel times.
- Operational policies (e.g., max transfers per job, hazardous/special item rules).

## Data Model Requirements (Target)

Existing tables already cover part of this (`jobs`, `shipments`, `trucks`, `workers`).
For full optimization, we should standardize:

- job-level demand:
  - required volume/weight
  - pickup/drop windows
  - split-allowed flag
- shipment leg model:
  - `from_location`, `to_location`, status, assigned truck(s)
- transfer events:
  - transfer node, inbound truck, outbound truck, timestamp, quantity
- route plans:
  - sequence number per stop,
  - ETA/ETD,
  - planned load after each stop.

## Recommended Planning Approach

Use a staged optimizer rather than a single monolithic solve:

0. **Planner interaction phase**
   - user selects roadway/corridor/site extents visually
   - historical overlap, route familiarity, and profitability context surface candidate legs
   - the system proposes a draft operational plan before low-level assignment begins
1. **Tender/job pre-ranking**
   - profitability + urgency + route familiarity + spare-capacity signals.
2. **Assignment phase**
   - choose feasible truck/crew combinations for each candidate job/leg.
3. **Sequencing phase**
   - optimize pickup/drop order for each truck route.
4. **Transfer phase (optional)**
   - evaluate whether cross-dock transfer improves objective enough to justify handling risk.
5. **Repair phase**
   - when disruptions occur, re-optimize locally with minimal plan churn.

## Heuristic vs Exact Solve

Short term (pragmatic):

- heuristics + local search (greedy insertion, 2-opt/3-opt style route improvement),
- deterministic rules for transfer eligibility and max handoffs.

Long term (advanced):

- MIP/CP-SAT for tighter global optimization on bounded planning horizons,
- scenario simulation for peak demand surges.

## Peak-Season Policy

Peak periods can justify temporary "beyond nominal capacity" operations when approved:

- hire additional trucks/drivers,
- subcontract overflow routes,
- permit higher transfer volume if SLA risk remains acceptable.

Policy guardrails:

- track margin after all overflow costs,
- cap transfer complexity per job,
- enforce hard safety/compliance limits even during peak load.

## Integration With Current Corkysoft Flows

This optimization concept should align with current components:

- Kent tender triage (`/kent-ams/tenders/prioritized`) for upfront focus.
- Quote workflow operational signal (`quote_operational_signals`) for day-of quoting context.
- Ingest workflow signal (`job_operational_signals`) so imported jobs carry operational fit metadata.

## V1 Policy Decisions

### Transfer Eligibility

Solver may recommend transfers only when:

- transfer occurs at an approved depot/cross-dock location
- no safety/legal/compliance rule is violated
- expected commercial outcome improves or a manager-approved capacity policy applies
- resulting plan remains explainable to an operator

Solver may not recommend transfers when:

- hazardous/special-item handling forbids it
- transfer count exceeds the configured job limit
- route/service risk rises without a documented commercial justification

### Job Split Rules

Job splitting is allowed only when:

- the job is explicitly marked split-eligible
- all resulting legs preserve pickup-before-delivery precedence
- the operator can see which load portion is on which truck

Job splitting is not allowed by default for:

- special-handling jobs
- jobs where customer/SLA rules require single-custody transport

### Explainability Requirements

Any future solver output must expose:

- why this plan was chosen
- which constraints were binding
- whether transfer or split rules were invoked
- expected margin/cost impact
- operator-visible risk flags

Planner interaction reference:
- [Planner Interaction Model](planner_interaction_model.md)

## Near-Term Documentation Deliverables

1. Define the planner interaction model for visually selecting roadway/corridor/site extents and surfacing historical overlap.
2. Define transfer policy and allowed nodes.
3. Define job split rules (what can be split, and when).
4. Define route-plan schema (stop sequence, load evolution, transfer markers).
5. Define operator UI expectations for explainability ("why this plan was chosen").
