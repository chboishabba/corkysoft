# Rollout Execution User Stories

This document defines the live rollout and spreadsheet-decommissioning stories
that remain after the native workflows are implemented. These stories are
separate from product-operation stories because they govern cutover execution,
fallback handling, and promotion between rollout states.

## Dispatcher

Trigger:
- daily dispatch runs from `Operations` -> `Dispatch`, but an external handoff
  or temporary fallback may still be needed

Primary decisions:
- can work continue from the native board
- does a snapshot need to be exported for an external team
- does the bypass of the native board need to be logged as fallback use

Required inputs:
- `Operations` -> `Dispatch` status
- dispatch workflow cutover status
- recent cutover events
- job and segment readiness

Expected system outputs:
- snapshot export when needed
- logged `snapshot_issued` event
- logged `fallback_use` event whenever native dispatch is bypassed

Operator actions:
- export a dispatch snapshot
- record fallback use with a reason
- continue from the native board unless blocked by a real exception

Success criteria:
- dispatchers do not need tribal knowledge to know current rollout state
- non-native dispatch actions are visible in the cutover history

## Fleet / Operations Manager

Trigger:
- scheduled rollout review
- fallback drill cadence
- repeated exceptions or fallback use

Primary decisions:
- is the workflow healthy enough to stay in its current cutover state
- should a fallback drill be run now
- is there enough evidence to request promotion

Required inputs:
- derived native usage metrics
- open issue count
- fallback-use history
- snapshot-consumer count
- checklist state
- recommendation and approval state

Expected system outputs:
- logged `review` event
- logged `fallback_drill` event
- logged `promotion_requested` event when promotion is warranted
- visible reason when a promotion is still blocked

Operator actions:
- record review
- record fallback drill
- request promotion with an explicit note
- maintain rollback instructions and checklist state

Success criteria:
- review and drill cadence are visible in-app
- promotion requests are evidence-driven, not anecdotal

## Commercial Owner

Trigger:
- an operations manager requests promotion for a workflow

Primary decisions:
- does the evidence justify promotion
- are open issues and fallback use acceptable
- are snapshot dependencies low enough to move forward

Required inputs:
- current workflow status
- target status
- recommendation reason
- target-met state
- fallback-use count
- recent review and drill events
- snapshot-consumer count

Expected system outputs:
- logged `promotion_approved` or `promotion_rejected` event
- preserved actor and note for the decision
- no status transition until approval exists

Operator actions:
- approve promotion
- reject promotion with a reason
- require another review or drill when evidence is weak

Success criteria:
- promotions are governed commercial decisions
- approval and rejection decisions are auditable later

## Rollout Coordinator / Admin

Trigger:
- a promotion is requested and the evidence gate is already satisfied

Primary decisions:
- has the ops review path completed
- has the commercial approval path completed
- should the transition be applied now

Required inputs:
- actionable recommendation
- latest request and approval state
- current workflow status

Expected system outputs:
- logged `status_transition` event
- updated workflow status
- preserved approval trail tied to the target status

Operator actions:
- inspect the approval chain
- apply the transition only after approval is complete
- refuse to apply transitions that are missing approval

Success criteria:
- no workflow status changes without a visible approval trail
- rollout state changes remain deliberate and reversible
