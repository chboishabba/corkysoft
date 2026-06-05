from __future__ import annotations

import requests

import pytest

from corkysoft.whisperx_adapter import (
    WhisperXAdapterError,
    fetch_task_status,
    submit_transcription,
)


class _FakeResponse:
    def __init__(
        self,
        payload=None,
        *,
        status_code: int = 200,
        text: str = "",
        json_error: Exception | None = None,
    ):
        self._payload = payload
        self.status_code = status_code
        self.text = text
        self._json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("http error", response=self)

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload


def test_submit_transcription_rejects_invalid_json(monkeypatch):
    monkeypatch.setenv("WHISPERX_BASE_URL", "http://whisperx.local")
    monkeypatch.setattr(
        "corkysoft.whisperx_adapter.requests.post",
        lambda *args, **kwargs: _FakeResponse(json_error=ValueError("bad json")),
    )

    with pytest.raises(WhisperXAdapterError, match="returned invalid JSON"):
        submit_transcription(
            service_key="ops",
            file_bytes=b"audio",
            filename="call.wav",
        )


def test_fetch_task_status_rejects_non_object_json(monkeypatch):
    monkeypatch.setenv("WHISPERX_BASE_URL", "http://whisperx.local")
    monkeypatch.setattr(
        "corkysoft.whisperx_adapter.requests.get",
        lambda *args, **kwargs: _FakeResponse(["not", "an", "object"]),
    )

    with pytest.raises(WhisperXAdapterError, match="non-object payload"):
        fetch_task_status(service_key="ops", identifier="task-1")


def test_submit_transcription_includes_http_error_body(monkeypatch):
    monkeypatch.setenv("WHISPERX_BASE_URL", "http://whisperx.local")
    monkeypatch.setattr(
        "corkysoft.whisperx_adapter.requests.post",
        lambda *args, **kwargs: _FakeResponse(
            status_code=500,
            text="adapter failed",
        ),
    )

    with pytest.raises(WhisperXAdapterError, match="HTTP 500: adapter failed"):
        submit_transcription(
            service_key="ops",
            file_bytes=b"audio",
            filename="call.wav",
        )
