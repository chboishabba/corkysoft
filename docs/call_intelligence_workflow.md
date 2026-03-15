# Call Intelligence Workflow

This document defines the implemented foundation for Corkysoft call intelligence.

## Purpose

Corkysoft now treats operational calls as first-class records rather than loose transcript files or operator notes.

The current foundation supports:

- `call_session` capture with routed `call_leg` history
- phone-based client auto-linking
- transcript artifact storage
- ambient office transcript sessions
- extracted action review and acceptance
- worker time-capture events on the same operational pipe
- append-only downstream event preparation for StatiBaker-style consumers

## Current State

Implemented:

- `Calls` dashboard tab
- API routes for call sessions, routed legs, routing events, legacy call events, transcript artifacts, ambient transcript sessions, worker time events, and append-only egress inspection
- WhisperX-WebUI adapter seam using its async task model (`/transcription/` plus `/task/{id}`)
- shared event substrate for both general ops calls and worker clock-on/off capture
- compatibility wrapper where legacy `call_event` creation becomes a one-session/one-leg case

Not implemented yet:

- live telephony / IVR ingestion
- WhatsApp ingestion
- automatic imperative extraction from transcript text
- live streaming transcript UI
- automatic StatenBaker push/delivery worker
- accommodation/provider integrations on top of call outcomes

## Operational Model

### 1. Call session first

Every operational phone interaction should land as a `call_session`.

The session is the parent business interaction.
Each routed or transferred step is a `call_leg`.

This supports:

- one-recipient calls
- call-centre routing
- operator to manager consults
- operator to worker/client follow-up legs
- shared-number timesheet branching

Routing itself is auditable through explicit routing events such as:

- `call_received`
- `call_routed`
- `call_answered`
- `call_transferred`
- `call_consult_started`
- `call_consult_ended`
- `call_ended`

Transcription begins on answered leg pickup, not at ringing.

### 2. Phone-based client linking

On session creation, Corkysoft attempts to normalize the caller/callee phone and match an existing client.

If no client matches, Corkysoft creates a placeholder client record so the call still has an operational owner.

### 3. Transcript artifacts are advisory

Transcript artifacts are stored separately from authoritative notes.

Current supported paths:

- fake transcript generation for workflow testing
- manual transcript artifact creation
- WhisperX task submission
- WhisperX task polling and artifact finalization
- transcript artifacts attached to specific answered legs
- separate ambient office transcript sessions

Authoritative rule:

- raw transcript is advisory
- extracted actions are advisory until accepted
- operator notes and accepted actions are authoritative operational state

### 4. Extracted actions

Call-linked extracted actions can now be:

- created
- reviewed
- accepted
- rejected

Accepted actions become the authoritative reviewed action trail for the call.

### 5. Worker time capture on the same event pipe

Worker clock-on / clock-off events now use the same operational intake model.

They support:

- app / WhatsApp / voice-call style channel tags
- phone-based worker matching where possible
- confidence scoring
- review queue semantics (`pending_review`, `accepted`, `rejected`)

### 6. Ambient office transcript sessions

Always-on office discussion capture is modeled separately from phone calls.

Use `ambient_session` for:

- office coordination talk
- manager/operator planning not tied to a handset leg
- off-call operational discussion that still needs transcript-backed review

This keeps ambient capture from being distorted into fake phone-call records while still letting accepted notes/actions flow back to operational truth.


## Fake Transcript Ingest Surface

The practical ingest surface for now is a fake-transcript generator.

Reasoning:

- it exercises the real Corkysoft call model without waiting for telephony
- it lets operators attach realistic-looking call outcomes to jobs, workers, and segments
- it keeps the implementation honest by forcing the review surfaces to work before live ASR exists

Current behavior:

- operator selects a call leg or ambient session
- operator provides an optional scenario note and desired outcome
- Corkysoft generates a no-timestamp transcript artifact linked to that leg or ambient session
- the generated transcript still remains advisory until notes or extracted actions are accepted

This should remain available even after live telephony exists because it is useful for:

- demos
- workflow testing
- backfilling calls that were handled outside the primary phone path
- training operators on the review surface

## WhisperX-WebUI Integration

Corkysoft does not run WhisperX locally for this feature.

Instead it uses an adapter to the external WhisperX-WebUI backend:

- submit audio to `/transcription/`
- store returned task identifier
- poll `/task/{identifier}`
- normalize result segments into transcript artifacts

Service separation is supported through config:

- `ops`
- `worker_time`

This matches the intended uptime split without hardwiring two codepaths into Corkysoft.

## StatiBaker Alignment

Corkysoft now produces append-only downstream-ready events for:

- call creation
- transcript availability
- note creation
- extracted action decisions
- link resolution
- worker time-capture record/review

Current state:

- events are stored locally in `state_egress_events`
- no delivery worker/push transport is implemented yet

## Dashboard Surface Ownership

### `Calls`

Primary use:

- operator review of call sessions, routed legs, transcript artifacts, accepted actions, ambient sessions, and worker-time events

Current subflows:

- create call session
- review legs and routing history
- resolve links
- generate fake transcripts for workflow testing
- submit/poll WhisperX tasks
- add notes
- accept/reject extracted actions
- create ambient office transcript sessions
- review worker time captures
- inspect append-only egress rows

### `Staff` / `Driver shifts`

Consume worker time capture results after review.

### `Dispatch` / `Operations`

Remain the operational truth for jobs and segments; calls attach context to those workflows rather than replacing them.

## Next Steps

1. add live telephony / IVR ingestion
2. add WhatsApp ingestion
3. add machine-extracted imperative suggestions from transcript text
4. add delivery worker for StatiBaker append-only egress
5. refine the live Calls Console around multi-hop manager/operator/crew/client escalation and accepted-action promotion


## StatiBaker Delivery Worker Thoughts

The right delivery-worker design is conservative.

Recommended shape:

- treat `state_egress_events` as a local outbox
- deliver in append-only order
- never mutate or delete successfully emitted rows during normal operation
- use idempotency keys and correlation ids exactly as stored

What I want:

- a small pull/push worker that can resume safely after crashes
- explicit delivery state rather than hidden retries
- batch delivery where useful, but single-event semantics must still be traceable
- clear boundary: Corkysoft remains operational truth, StatiBaker remains downstream reduction/distillation

What I do not want:

- direct synchronous delivery from the operator UI path
- hidden best-effort fire-and-forget semantics
- transcript text being mistaken for authoritative state simply because it was emitted
- StatiBaker becoming a second mutable workflow database

Practical implementation notes:

- add a delivery receipt table or delivery-attempt columns rather than overloading the event rows themselves with too much mutable state
- keep authority-class distinctions intact (`compiled_state` vs `observer_capture_ref`)
- deliver accepted actions and operator notes as stronger signals than raw transcript artifacts
- make the worker restartable and auditable before making it automatic

The first safe milestone is:

1. background/manual worker reads undelivered rows
2. posts append-only envelopes to the StatiBaker ingress seam
3. records success/failure receipts locally
4. exposes a small admin review surface for stuck deliveries

That will be enough to prove the integration without turning Corkysoft into delivery middleware too early.
