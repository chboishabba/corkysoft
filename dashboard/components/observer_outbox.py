"""Formatting helpers for observer outbox rows.

These helpers are intentionally pure so the dashboard can reuse them in a
compact table view and a richer detail view without duplicating JSON decoding
or reference formatting logic.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


def decode_observer_json(value: Any, default: Any) -> Any:
    """Decode a JSON-like value or return *default* when empty/invalid."""

    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return default
    return value


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def _compact_ref_string(refs: Any) -> str:
    if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
        return ""
    parts: list[str] = []
    for ref in refs:
        if not isinstance(ref, Mapping):
            continue
        kind = str(ref.get("kind") or "ref")
        label = ref.get("table") or ref.get("rowId") or ref.get("eventId") or ref.get("id")
        if label is None and ref:
            label = next(iter(ref.values()))
        if label is None:
            parts.append(kind)
        else:
            parts.append(f"{kind}:{label}")
    return ", ".join(parts)


def _compact_mapping_string(mapping: Mapping[str, Any], *, max_items: int = 5) -> str:
    items = []
    for key in sorted(mapping.keys(), key=str):
        value = mapping[key]
        if value in (None, "", [], {}):
            continue
        if isinstance(value, Mapping):
            value_text = _compact_mapping_string(value, max_items=max_items)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            value_text = ", ".join(str(item) for item in value[:max_items])
            if len(value) > max_items:
                value_text += f" +{len(value) - max_items}"
        else:
            value_text = str(value)
        items.append(f"{key}={value_text}")
    if not items:
        return ""
    visible = items[:max_items]
    suffix = "" if len(items) <= max_items else f" +{len(items) - max_items}"
    return "; ".join(visible) + suffix


def _payload_key_summary(payload: Any, *, max_keys: int = 5) -> str:
    if not isinstance(payload, Mapping):
        return ""
    keys = sorted(str(key) for key in payload.keys())
    if not keys:
        return ""
    visible = keys[:max_keys]
    suffix = "" if len(keys) <= max_keys else f" +{len(keys) - max_keys}"
    return ", ".join(visible) + suffix


def normalize_observer_event_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized observer-event row with decoded nested fields."""

    object_refs = decode_observer_json(_first_present(row, "objectRefs", "object_refs_json"), {})
    provenance_refs = decode_observer_json(
        _first_present(row, "provenanceRefs", "provenance_refs_json"),
        [],
    )
    evidence_refs = decode_observer_json(
        _first_present(row, "evidenceRefs", "evidence_refs_json"),
        [],
    )
    payload = decode_observer_json(_first_present(row, "payload", "payload_json"), {})

    if not isinstance(object_refs, Mapping):
        object_refs = {}
    if not isinstance(payload, Mapping):
        payload = {}
    if not isinstance(provenance_refs, list):
        provenance_refs = list(provenance_refs) if isinstance(provenance_refs, Sequence) else []
    if not isinstance(evidence_refs, list):
        evidence_refs = list(evidence_refs) if isinstance(evidence_refs, Sequence) else []

    job_id = _first_present(object_refs, "job_id", "jobId")
    if job_id is None and "job_ids" in object_refs:
        job_ids = object_refs.get("job_ids")
        if isinstance(job_ids, Sequence) and not isinstance(job_ids, (str, bytes)) and len(job_ids) == 1:
            job_id = job_ids[0]
    if job_id is None:
        job_id = _first_present(payload, "jobId", "job_id")

    event_time = _first_present(row, "eventTime", "event_time", "occurredAt")
    recorded_at = _first_present(row, "recordedAt", "recorded_at", "ingestedAt")

    normalized = dict(row)
    normalized["eventId"] = _first_present(row, "eventId", "event_id")
    normalized["eventFamily"] = _first_present(row, "eventFamily", "event_family")
    normalized["eventType"] = _first_present(row, "eventType", "event_type")
    normalized["actorRef"] = _first_present(row, "actorRef", "actor_ref")
    normalized["authorityClass"] = _first_present(row, "authorityClass", "authority_class")
    normalized["summary"] = _first_present(row, "summary")
    normalized["status"] = _first_present(row, "status")
    normalized["sourceEntityId"] = _first_present(row, "sourceEntityId", "source_entity_id")
    normalized["payloadHash"] = _first_present(row, "payloadHash", "payload_hash")
    normalized["objectRefs"] = object_refs
    normalized["provenanceRefs"] = provenance_refs
    normalized["evidenceRefs"] = evidence_refs
    normalized["payload"] = payload
    normalized["jobId"] = job_id
    normalized["eventTime"] = event_time
    normalized["recordedAt"] = recorded_at
    normalized["compactObjectRefs"] = _compact_mapping_string(object_refs)
    normalized["compactProvenanceRefs"] = _compact_ref_string(provenance_refs)
    normalized["compactEvidenceRefs"] = _compact_ref_string(evidence_refs)
    normalized["payloadKeySummary"] = _payload_key_summary(payload)
    return normalized


def build_observer_event_table_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return compact rows for a table view."""

    table_rows: list[dict[str, Any]] = []
    for row in rows:
        event = normalize_observer_event_row(row)
        table_rows.append(
            {
                "Event time": event.get("eventTime"),
                "Family": event.get("eventFamily"),
                "Type": event.get("eventType"),
                "Actor": event.get("actorRef"),
                "Authority": event.get("authorityClass"),
                "Status": event.get("status"),
                "Job": event.get("jobId"),
                "Summary": event.get("summary"),
                "Object refs": event.get("compactObjectRefs"),
                "Provenance": event.get("compactProvenanceRefs"),
                "Payload keys": event.get("payloadKeySummary"),
            }
        )
    return table_rows


def build_observer_event_detail_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return a key/value detail view for one observer event."""

    event = normalize_observer_event_row(row)
    return [
        {"Field": "Event ID", "Value": event.get("eventId")},
        {"Field": "Event time", "Value": event.get("eventTime")},
        {"Field": "Recorded at", "Value": event.get("recordedAt")},
        {"Field": "Source entity", "Value": event.get("sourceEntityId")},
        {"Field": "Family", "Value": event.get("eventFamily")},
        {"Field": "Type", "Value": event.get("eventType")},
        {"Field": "Actor", "Value": event.get("actorRef")},
        {"Field": "Authority", "Value": event.get("authorityClass")},
        {"Field": "Status", "Value": event.get("status")},
        {"Field": "Job", "Value": event.get("jobId")},
        {"Field": "Summary", "Value": event.get("summary")},
        {"Field": "Object refs", "Value": event.get("objectRefs")},
        {"Field": "Provenance refs", "Value": event.get("provenanceRefs")},
        {"Field": "Evidence refs", "Value": event.get("evidenceRefs")},
        {"Field": "Payload", "Value": event.get("payload")},
        {"Field": "Payload hash", "Value": event.get("payloadHash")},
    ]


__all__ = [
    "build_observer_event_detail_rows",
    "build_observer_event_table_rows",
    "decode_observer_json",
    "normalize_observer_event_row",
]
