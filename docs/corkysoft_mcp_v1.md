# Corkysoft MCP v1 Contract

This document defines the first MCP adapter posture for Corkysoft.

It is intentionally narrower than the broader SB / ITIR downstream contract:

- this is an adapter contract for querying Corkysoft capabilities
- it does not make MCP the owner of Corkysoft workflow truth
- it starts read-only and deterministic

## Boundary

- Corkysoft remains authoritative for removals workflow state, pricing inputs,
  planning, dispatch recommendations, diary review, and reconciliation state.
- An MCP adapter may expose bounded Corkysoft read models and advisory outputs.
- StatiBaker remains a downstream reviewed-state consumer, not a second
  operational cockpit.
- ITIR remains the orchestration/context and contract-hygiene layer across
  systems.
- MCP must not redefine, fork, or silently upgrade Corkysoft business
  semantics.

## v1 Goals

- expose a small read-only integration surface over already-implemented
  Corkysoft logic
- make those tools deterministic enough for agent and automation use
- preserve producer ownership in the existing analytics and API layers
- keep the transport replaceable without changing result semantics

## v1 Non-Goals

Do not use v1 to expose:

- mutable dispatch or Kent-admin actions
- policy writes or approval actions
- hidden admin-only workflow state without existing Corkysoft governance
- raw database tables as an unbounded query surface
- autonomous pricing, dispatch, or reconciliation control

## Adapter Shape

The preferred shape is a dedicated adapter package, for example:

- `corkysoft/mcp/`
- or a sibling package such as `corkysoft-mcp`

The adapter should wrap existing producer-owned helpers instead of duplicating
business logic. MCP is an integration layer, not a second source of truth.

## Result Envelope

All tools should return a stable envelope:

### Success

```json
{
  "ok": true,
  "result": {}
}
```

### Failure

```json
{
  "ok": false,
  "error": {
    "code": "string",
    "message": "string",
    "details": {}
  }
}
```

Rules:

- `result` should be deterministic for the same inputs and DB state
- `error.code` should be stable enough for client handling
- tool output should prefer Corkysoft-native identifiers and reviewed summaries
  over denormalized raw table dumps

## First Tool Family

The first MCP slice should stay read-only and query-oriented:

- `corkysoft.profitability_summary`
  - route, corridor, and margin summary over the existing analytics layer
- `corkysoft.dispatch_recommendations`
  - current share/reallocation and utilization-response recommendations
- `corkysoft.operations_diary_summary`
  - day/week/job review summary and exception state
- `corkysoft.quote_guidance_preview`
  - benchmark and quote-guidance preview without mutating quote state

These are good v1 candidates because they already exist as bounded Corkysoft
surfaces and can be exposed without inventing new workflow semantics.

## Deferred Tool Families

The following should stay out of v1:

- dispatch mutation tools
- Kent admin policy or reason-code writes
- observer-outbox delivery control
- approval-chain or rollout state changes
- auth-sensitive governance actions not already formalized for external use

## Transport Posture

The contract should be transport-agnostic.

Likely viable transport shapes:

- JSON-line local bridge for local/agent use
- FastMCP-style local adapter
- a persistent JSON bridge when long-lived orchestration needs it

The transport may change later. The result envelope and tool semantics should
not.

## Current Implementation Status

The first implementation slice now exists locally:

- `corkysoft/mcp/registry.py`
  - namespaced tool registry
- `corkysoft/mcp/tools.py`
  - four read-only tool adapters over existing Corkysoft helpers
- `corkysoft/mcp/bridge.py`
  - working JSON-line local bridge and supported default entrypoint
- `corkysoft/mcp/server.py`
  - optional FastMCP transport when the Python MCP SDK is installed
- `corkysoft/mcp/__main__.py`
  - defaults to the JSON bridge and exposes explicit `--bridge` / `--server`
    CLI selection

The contract is therefore no longer docs-only. The local bridge and registry
are implemented, but transport hardening and any mutable tool posture remain
deferred.

## Governance

Promotion criteria for the first implementation:

- documented tool contract
- stable namespacing
- read-only posture
- tests for registry, parameter validation, and result-envelope behavior
- one working local transport path

Do not promote mutable tools until Corkysoft auth, audit, and operator-policy
boundaries are strong enough to govern them.

## Relationship To Existing Cross-Project Docs

- `docs/sb_itir_downstream_contract.md` covers reviewed downstream export
  envelopes for planner/diary/reconciliation output.
- this document covers interactive MCP tool exposure over Corkysoft-owned
  read models
- the two contracts are complementary, not interchangeable

## Ordered Next Steps

1. Keep the current bridge-default CLI and registry stable while validating more real DB states.
2. Add explicit tests for envelope stability and producer-boundary discipline across more seeded scenarios.
3. Decide whether the optional FastMCP stdio server should stay opt-in or become a separately supported transport tier.
4. Reassess whether any mutable workflow tools are governable enough for a v2.
