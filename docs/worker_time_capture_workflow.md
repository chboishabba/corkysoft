# Worker Time Capture Workflow

This document defines how Corkysoft should capture worker clock-on and
clock-off events across mixed communication environments, including remote work
where smartphones or apps cannot be assumed.

Use this as the canonical planning document for workforce time capture before
implementation.

## Purpose

Time capture needs to work for:

- staff with smartphones
- staff with WhatsApp only
- staff with basic mobile phones
- remote workers calling from landlines or site phones

The system should optimize for reliable capture first, convenience second.

## Core Requirements

- capture worker identity
- capture clock-on / clock-off event type
- capture effective event timestamp
- capture source channel
- capture confidence and auditability
- tolerate low-tech channels
- avoid requiring spreadsheet-side timesheet truth

## Supported Input Channels

### 1. Native app / dashboard

Use when:

- worker or supervisor has direct app access

Strengths:

- highest structure
- lowest ambiguity
- immediate validation against roster / assignment / segment context

Weaknesses:

- cannot be the only capture path

### 2. WhatsApp-assisted capture

Use when:

- workers can send text/voice/photo through WhatsApp but do not have the main
  app available

Expected flow:

- worker sends clock-on or clock-off message
- intake layer extracts worker identifier, event type, and time
- ambiguous messages are flagged for supervisor review

Strengths:

- familiar to crews
- lower friction than forcing app installs

Weaknesses:

- semi-structured
- requires parsing and confidence handling

### 3. Phone / landline call-in

Use when:

- worker is remote
- worker has only voice access
- site conditions make typed input unrealistic

Expected flow:

- worker calls a dedicated number
- phone tree / IVR prompts for:
  - employee number
  - clock-on or clock-off
  - optional job / truck / site code
  - event time if not “now”
- audio is stored or transcribed
- transcript parser extracts the event
- low-confidence captures are queued for human review

Strengths:

- works even from landlines
- does not assume smartphone ownership

Weaknesses:

- speech ambiguity
- stronger audit and review requirements

## Event Model

Each captured event should store:

- worker identifier
- event type: `clock_on` or `clock_off`
- captured timestamp
- effective timestamp
- channel: `app`, `whatsapp`, `voice_call`, `manual_supervisor`
- optional related job / segment / truck
- optional phone number / caller id
- transcription or raw message payload where relevant
- confidence score
- review status
- reviewer / correction audit when manually fixed

## Validation Rules

Minimum validation:

- worker must resolve to a known worker record or a review queue item
- event type must be explicit or confidently inferred
- duplicate rapid-fire events should be flagged
- events outside expected shift windows should warn, not silently fail

## Operational Model

### Routine path

- worker submits event through any supported channel
- system matches worker and time context
- event is accepted if confidence is high

### Review path

- low-confidence parse
- unknown caller or employee number
- implausible timestamp
- duplicate or contradictory event

Supervisor or labor planner reviews and resolves the record.

## Relationship to Planning

Time capture should integrate with:

- `Operations` -> `Staff`
- `Operations` -> `Driver shifts`
- `Operations`
- `Operations` -> `Dispatch`

Primary uses:

- roster actuals vs plan
- labor-cost rollup
- attendance visibility
- no-show / late-start detection

## Recommended Implementation Order

1. manual structured event entry
2. WhatsApp text/voice ingestion
3. phone/landline IVR or call transcription
4. confidence scoring and review queue
5. payroll / timesheet exports if needed later

## Notes

- The phone / landline path is a first-class requirement, not a fallback afterthought.
- Voice capture should be designed around supervised operational accuracy, not around pretending ASR is perfect.
