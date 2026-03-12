# AMS Backend Documentation Playbook

This playbook captures a practical structure for documenting a relocation-style
Assignee Management System (AMS), adapted for Corkysoft's logistics and pricing
domain.

Use it as a documentation template when onboarding engineers or formalizing
underdocumented modules.

## 1. Recommended Documentation Structure

1. System overview
   - Purpose and scope
   - Core concepts and terminology
2. Architecture
   - High-level diagram
   - Service/module boundaries
   - Data flow and external dependencies
3. Data model
   - Entity list
   - Relationships
   - Status enums/state transitions
   - Schema reference
4. Workflows
   - End-to-end move/quote lifecycle
   - Exception paths (failed routing, missing geocode, stale telemetry)
5. API and interfaces
   - Authentication/authorization assumptions
   - Endpoint/CLI signatures and payloads
   - Error contracts
6. Integrations
   - Routing providers
   - Fleet/dispatch/accounting/CRM connectors
   - Import/export contracts
7. Operations
   - Deployment and runtime configuration
   - Monitoring and logging
   - Debugging and recovery notes
8. Runbooks
   - Create/price a job
   - Investigate bad margin outcomes
   - Reprocess failed imports

## 2. Reference Entity Model For Relocation Backends

Typical relocation backends center on these entities:

- `assignees` (or customers/clients)
- `moves` (or jobs)
- `shipments`
- `services`
- `vendors`
- `inventory_items`
- `documents`
- `tasks`

For Corkysoft, map these concepts to existing objects in `routes.db` and
dashboard-facing analytics outputs, then keep a glossary of equivalent names.

## 3. Reverse-Engineering Checklist

When docs are incomplete, use this order:

1. Identify core entities and tables.
2. Enumerate status fields and allowed transitions.
3. Trace one real record across create -> process -> close.
4. Inspect API/CLI surfaces and payload contracts.
5. List external integrations and handoff points.
6. Identify event pipelines/background jobs if present.

## 4. Minimum Diagram Set

Keep one simple lifecycle diagram and one entity relationship diagram current at
all times.

Example lifecycle (adapt to local terminology):

`Job Created -> Survey Scheduled -> Quote Approved -> Work Scheduled -> In Transit -> Delivered -> Closed`

## 5. Documentation Priority For This Repo

Prioritize these pages in order:

1. Move/job lifecycle mapped to current CLI + dashboard flows.
2. Data model glossary aligning relocation terms to Corkysoft tables/fields.
3. Pricing engine factors and how each input affects margin outputs.

This sequence reduces onboarding time while making analytics assumptions explicit.
