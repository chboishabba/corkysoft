from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Mapping
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from urllib.parse import parse_qs, urlparse

import pytest

from corkysoft.importers import Job, JobsImportError, JobsImporter


def _build_payload(job_id: str, **overrides: Any) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "id": job_id,
        "name": f"Job {job_id}",
        "status": "active",
        "jobType": "Move",
        "method": "Road",
        "origin": "Brisbane",
        "destination": "Sydney",
        "created": "2025-11-13T23:10:20.888Z",
        "jobDate": "2025-11-13T23:10:20.888Z",
        "salesRep": "Alice",
        "moveManager": "Bob",
        "accountManager": "Charlie",
        "corporateContact": "Delta",
        "uplift": "2025-11-14T08:00:00Z",
        "delivery": "2025-11-15T08:00:00Z",
        "survey": "2025-11-10T09:00:00Z",
    }
    payload.update(overrides)
    return payload


def test_fetch_page_builds_expected_request() -> None:
    captured: dict[str, Any] = {}

    def fake_get(url: str, headers: Mapping[str, str]) -> Mapping[str, Any]:
        captured["url"] = url
        captured["headers"] = dict(headers)
        return {
            "jobs": [_build_payload("123")],
            "_links": {"self": {"href": url}},
        }

    importer = JobsImporter(
        "https://api.example.com/v1",
        token="secret",
        http_get=fake_get,
    )

    jobs, links = importer.fetch_page(
        offset=5,
        limit=25,
        filters={"status": "scheduled", "flexible": True, "origin_phone": "12345"},
    )

    assert isinstance(links, Mapping)
    assert len(jobs) == 1
    job = jobs[0]
    assert job.id == "123"
    assert job.created == datetime(2025, 11, 13, 23, 10, 20, 888000, tzinfo=UTC)
    assert captured["headers"] == {
        "Accept": "application/json",
        "Authorization": "Bearer secret",
    }

    parsed = urlparse(captured["url"])
    assert parsed.scheme == "https"
    assert parsed.netloc == "api.example.com"
    assert parsed.path == "/v1/jobs"

    params = parse_qs(parsed.query)
    assert params["offset"] == ["5"]
    assert params["limit"] == ["25"]
    assert params["status"] == ["scheduled"]
    assert params["flexible"] == ["true"]
    assert params["originPhone"] == ["12345"]


def test_fetch_all_iterates_until_no_more_results() -> None:
    calls: list[str] = []

    def fake_get(url: str, headers: Mapping[str, str]) -> Mapping[str, Any]:
        calls.append(url)
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        offset = int(params.get("offset", [0])[0])
        limit = int(params.get("limit", [2])[0])
        jobs = []
        for index in range(limit):
            job_index = offset + index
            if job_index >= 3:
                break
            jobs.append(
                _build_payload(str(job_index), created=f"2025-11-1{job_index}T01:00:00Z")
            )
        return {"jobs": jobs}

    importer = JobsImporter("https://api.example.com", http_get=fake_get, default_limit=2)

    jobs = list(importer.fetch_all(limit=2))

    assert [job.id for job in jobs] == ["0", "1", "2"]
    assert len(calls) == 2
    assert all(isinstance(job, Job) for job in jobs)


def test_fetch_page_validates_response_structure() -> None:
    importer = JobsImporter("https://api.example.com", http_get=lambda url, headers: {"jobs": {}})

    with pytest.raises(JobsImportError):
        importer.fetch_page()


def test_fetch_page_rejects_unknown_filter() -> None:
    importer = JobsImporter("https://api.example.com", http_get=lambda url, headers: {"jobs": []})

    with pytest.raises(ValueError, match="Unsupported filter"):
        importer.fetch_page(filters={"unsupported": "value"})


def test_fetch_all_rejects_invalid_limit() -> None:
    importer = JobsImporter("https://api.example.com", http_get=lambda url, headers: {"jobs": []})

    with pytest.raises(ValueError):
        list(importer.fetch_all(limit=0))
