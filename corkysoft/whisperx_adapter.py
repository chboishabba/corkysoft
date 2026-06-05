from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


class WhisperXAdapterError(RuntimeError):
    """Raised when the WhisperX backend cannot complete a request."""


@dataclass(frozen=True)
class WhisperXServiceConfig:
    service_key: str
    base_url: str
    timeout_seconds: int = 30


_SERVICE_ENV_PREFIX = {
    "ops": "WHISPERX_OPS",
    "worker_time": "WHISPERX_WORKER_TIME",
}


def resolve_whisperx_service_config(service_key: str) -> WhisperXServiceConfig:
    normalized = (service_key or "ops").strip().lower()
    if normalized not in _SERVICE_ENV_PREFIX:
        raise WhisperXAdapterError(f"Unsupported WhisperX service key: {service_key}")
    prefix = _SERVICE_ENV_PREFIX[normalized]
    base_url = (
        os.environ.get(f"{prefix}_BASE_URL")
        or os.environ.get("WHISPERX_BASE_URL")
        or ""
    ).strip()
    if not base_url:
        raise WhisperXAdapterError(
            f"WhisperX base URL is not configured for service '{normalized}'"
        )
    timeout_raw = (
        os.environ.get(f"{prefix}_TIMEOUT_SECONDS")
        or os.environ.get("WHISPERX_TIMEOUT_SECONDS")
        or "30"
    )
    try:
        timeout = max(int(timeout_raw), 1)
    except ValueError:
        timeout = 30
    return WhisperXServiceConfig(service_key=normalized, base_url=base_url.rstrip("/"), timeout_seconds=timeout)


def _request_error_message(action: str, exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return f"WhisperX {action} failed: {exc}"
    status = getattr(response, "status_code", None)
    text = (getattr(response, "text", "") or "").strip()
    if text:
        return f"WhisperX {action} failed with HTTP {status}: {text[:500]}"
    return f"WhisperX {action} failed with HTTP {status}: {exc}"


def _json_payload(response: requests.Response, *, action: str) -> Dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise WhisperXAdapterError(f"WhisperX {action} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise WhisperXAdapterError(f"WhisperX {action} returned a non-object payload")
    return payload


def submit_transcription(
    *,
    service_key: str,
    file_bytes: bytes,
    filename: str,
    language: Optional[str] = None,
    diarize: bool = True,
) -> Dict[str, Any]:
    config = resolve_whisperx_service_config(service_key)
    files = {"file": (filename, file_bytes, "application/octet-stream")}
    data: dict[str, Any] = {}
    if language:
        data["lang"] = language
    data["is_diarize"] = "true" if diarize else "false"
    try:
        response = requests.post(
            f"{config.base_url}/transcription/",
            files=files,
            data=data,
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network exercised via mocks/tests
        raise WhisperXAdapterError(_request_error_message("submission", exc)) from exc
    payload = _json_payload(response, action="submission")
    identifier = payload.get("identifier")
    if not identifier:
        raise WhisperXAdapterError("WhisperX submission returned no identifier")
    return payload



def fetch_task_status(*, service_key: str, identifier: str) -> Dict[str, Any]:
    config = resolve_whisperx_service_config(service_key)
    try:
        response = requests.get(
            f"{config.base_url}/task/{identifier}",
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network exercised via mocks/tests
        raise WhisperXAdapterError(_request_error_message("status poll", exc)) from exc
    return _json_payload(response, action="status poll")
