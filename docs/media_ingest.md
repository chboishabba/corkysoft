# Media ingest workflow (PEC photos & bodycam clips)

This note captures how Pre-Existing Condition (PEC) photos and event-based bodycam clips move from capture to durable storage and back into the `movement_events` timeline. The goal is to make evidence ingest predictable, hash-verifiable, and easy to reference alongside items handled on a job.

## Capture triggers
- **PEC photo set (mobile)**: triggered before loading begins, after unloading, or when the crew records damage. The mobile app prompts for a minimum front/back/wide set and enforces photo count before allowing the job state to advance.
- **Bodycam clip (event-based)**: triggered by belt accelerometer spike, RFID checkpoint scan, panic button, or manual tap inside the app. Clips are short (e.g., 20–60 seconds) and share the same job + movement event context as PEC stills.
- **RFID/scan-driven prompts**: if a pallet, box, or room crossing is detected via RFID/QR scan, the app prompts for a PEC still or bodycam bookmark and pre-fills the event metadata.

## Metadata requirements
Every upload payload must include:
- `job_id` and `movement_event_id` (load start, door seal, arrival, etc.) so media can anchor to the operational timeline.
- `item_id`/`asset_tag` when the capture is item-specific (e.g., a tagged piano or crate).
- Capture `timestamp` in UTC (RFC 3339), `device_id`, `captured_by` (user id), and `role` (driver, offsider, supervisor).
- Location metadata: `lat`, `lon`, optional `geofence_id` when near a depot/customer zone.
- `capture_type` (`pec_photo`, `bodycam_clip`) and `trigger` (`manual`, `rfid_scan`, `impact`, `panic_button`).
- Client privacy flags: `faces_present`, `customer_visible`, consent toggle for homeowner visibility.
- Integrity fields: SHA-256 hash of the raw file, byte length, MIME type, and original filename.
- Storage hints: target bucket/prefix, retention policy (`pec` vs `incident`), and optional `correlation_id` to group a burst of shots/clips.

## Hashing and timestamping
- Hash the file **on-device** before upload (SHA-256) and include the hex digest in the manifest.
- The API re-computes the hash server-side; mismatches reject the upload and leave the client to retry.
- Server writes an immutable `captured_at` (from client) and `ingested_at` (server clock) to support clock-skew audits.
- Optional signed receipt: return a manifest with the digest, server timestamp, storage URI, and user/device identifiers for insurer/legal use.

## Upload flow
1. **Capture & queue**: the app saves the raw file locally with the metadata manifest (JSON) and computed hash. Offline-first; queue persists until network is available.
2. **Upload**: multipart upload (file + manifest). The API validates fields, verifies hash, and normalises timestamps.
3. **Durable store**: write to object storage (e.g., `s3://corkysoft-audit/pec/<job_id>/` for stills, `.../bodycam/` for clips). The API stores only the signed URL/path, not the binary, in the primary database.
4. **Database linkage**: create/update `media` records keyed by `media_id` with foreign keys to `movement_events` and optional `items`/`assets` tables, including `hash`, `ingested_at`, `capture_type`, and `storage_uri`.
5. **Acknowledgement**: return `media_id`, resolved `storage_uri`, and the confirmed digest so the client can mark the queue item delivered.

## Storage layout and retention
- **Object storage prefixes**: `pec_photo` → `pec/<job_id>/<movement_event_id>/<timestamp>_<hash>.jpg`; `bodycam_clip` → `bodycam/<job_id>/<movement_event_id>/<timestamp>_<hash>.mp4`.
- **Cold-line/archival rules**: PEC stills retain for the move + claims window; incident clips can move to cheaper storage after N days with a manifest left hot for lookup.
- **Access control**: signed URLs limited to investigators/ops leads; RBAC ties back to the user role on the media row.

## Linkage to movement events and items
- Each `media` row references a `movement_event_id` so UI timelines can render thumbnails/bookmarks inline with scans and status changes.
- `item_id`/`asset_tag` links keep PEC evidence bound to specific inventory lines; if absent, the media is treated as job-level context.
- Queries should include a `correlation_id` to group a burst (e.g., pre-load PEC set), and expose hashes/timestamps so auditors can cross-check integrity.
- When a movement event is updated (e.g., corrected arrival time), the `media` rows remain immutable; the event table holds the revised timing while the digest + storage URI stay fixed.

## Stub: video/call processing roadmap
- **Video detection (Frigate)**: detect objects/rooms and generate event notes so staff can see what's present and where; register derived clips into the same media pipeline.
- **Call transcripts (Whisper)**: capture packing/move details from calls and attach conversation notes to the job timeline for quick review.
- **Spatial model (SFM)**: build a 3D house model to estimate access constraints and drive waypoint navigation/optimal pathing.

## Pseudocode: transcript-guided video processing pipeline
```text
inputs:
  - call_audio (job_id, device_id, captured_at)
  - video_streams (job_id, camera_id, start_ts, end_ts)
  - movement_events (job_id, event_ts, event_type)

process:
  1) transcript = whisper.transcribe(call_audio)
  2) segments = extract_segments(transcript)
     - each segment has (start_ts, end_ts, text, tags)
  3) timeline = build_timeline(segments, movement_events)
     - align transcript segments to job timeline and call start time
  4) for each segment in timeline:
       - derive target_window = map_to_video_window(segment, video_streams)
       - hints = parse_object_hints(segment.text)
       - frigate.request_detect(video_streams, target_window, hints)
  5) detections = collect_frigate_events(job_id)
  6) notes = build_staff_notes(segments, detections)
     - include "what/where" inventory cues + related timestamps
  7) persist:
       - media rows for derived clips + detection metadata
       - transcript_notes rows for staff review
       - cross-links: note_id <-> media_id <-> movement_event_id

outputs:
  - staff-facing notes with timestamps and locations
  - detection events with linked transcript excerpts
  - derived clips stored in media pipeline
```
