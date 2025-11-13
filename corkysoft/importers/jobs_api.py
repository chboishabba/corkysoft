"""Importer for the external jobs REST API."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


ResponsePayload = str | bytes | Mapping[str, Any]
HttpGet = Callable[[str, Mapping[str, str]], ResponsePayload]


def _parse_datetime(value: Any) -> datetime | None:
    """Parse ISO 8601 timestamps returned by the API."""

    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.endswith("Z"):
        cleaned = f"{cleaned[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


@dataclass(slots=True)
class Job:
    """Representation of a job returned by the external API."""

    id: str
    name: str | None = None
    status: str | None = None
    job_type: str | None = None
    method: str | None = None
    origin: str | None = None
    destination: str | None = None
    created: datetime | None = None
    job_date: datetime | None = None
    sales_rep: str | None = None
    move_manager: str | None = None
    account_manager: str | None = None
    corporate_contact: str | None = None
    uplift: datetime | None = None
    delivery: datetime | None = None
    survey: datetime | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_api(cls, payload: Mapping[str, Any]) -> "Job":
        """Create a :class:`Job` from the API response payload."""

        job_id = str(payload.get("id"))
        if not job_id:
            raise JobsImportError("Job payload is missing an id field")
        return cls(
            id=job_id,
            name=payload.get("name"),
            status=payload.get("status"),
            job_type=payload.get("jobType"),
            method=payload.get("method"),
            origin=payload.get("origin"),
            destination=payload.get("destination"),
            created=_parse_datetime(payload.get("created")),
            job_date=_parse_datetime(payload.get("jobDate")),
            sales_rep=payload.get("salesRep"),
            move_manager=payload.get("moveManager"),
            account_manager=payload.get("accountManager"),
            corporate_contact=payload.get("corporateContact"),
            uplift=_parse_datetime(payload.get("uplift")),
            delivery=_parse_datetime(payload.get("delivery")),
            survey=_parse_datetime(payload.get("survey")),
            raw=dict(payload),
        )


class JobsImportError(RuntimeError):
    """Raised when the jobs importer encounters an unrecoverable error."""


class JobsImporter:
    """Importer for the ``GET /jobs`` endpoint."""

    _FILTER_FIELDS = [
        "status",
        "externalId",
        "updatedBefore",
        "updatedAfter",
        "createdBefore",
        "createdAfter",
        "originPhone",
        "originMobile",
        "destinationPhone",
        "destinationMobile",
        "allPhone",
        "flexible",
        "originEmail",
        "destinationEmail",
        "allEmail",
    ]

    _FILTER_ALIASES = {
        "external_id": "externalId",
        "updated_before": "updatedBefore",
        "updated_after": "updatedAfter",
        "created_before": "createdBefore",
        "created_after": "createdAfter",
        "origin_phone": "originPhone",
        "origin_mobile": "originMobile",
        "destination_phone": "destinationPhone",
        "destination_mobile": "destinationMobile",
        "all_phone": "allPhone",
        "origin_email": "originEmail",
        "destination_email": "destinationEmail",
        "all_email": "allEmail",
    }

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        default_limit: int = 50,
        timeout: float = 10.0,
        http_get: HttpGet | None = None,
    ) -> None:
        if default_limit <= 0:
            raise ValueError("default_limit must be positive")
        self._base_url = base_url.rstrip("/") + "/"
        self._token = token
        self._default_limit = int(default_limit)
        self._timeout = float(timeout)
        self._http_get = http_get or self._default_http_get

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def default_limit(self) -> int:
        return self._default_limit

    def fetch_page(
        self,
        *,
        offset: int = 0,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> tuple[list[Job], Mapping[str, Any]]:
        """Fetch a single page of jobs from the API."""

        if offset < 0:
            raise ValueError("offset must be non-negative")
        page_size = limit if limit is not None else self._default_limit
        if page_size <= 0:
            raise ValueError("limit must be positive")

        query_pairs: list[tuple[str, str]] = [("offset", str(int(offset))), ("limit", str(int(page_size)))]
        if filters:
            query_pairs.extend(self._normalise_filters(filters))

        url = self._build_url("jobs", query_pairs)
        payload = self._perform_request(url)
        jobs_payload = payload.get("jobs")
        if not isinstance(jobs_payload, list):
            raise JobsImportError("Jobs API response did not contain a jobs list")

        jobs = [Job.from_api(item) for item in jobs_payload]
        links = payload.get("_links")
        if isinstance(links, Mapping):
            return jobs, links
        return jobs, {}

    def fetch_all(
        self,
        *,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> Iterator[Job]:
        """Yield jobs by paging through the API until exhaustion."""

        offset = 0
        page_size = limit if limit is not None else self._default_limit
        if page_size <= 0:
            raise ValueError("limit must be positive")

        while True:
            jobs, _ = self.fetch_page(offset=offset, limit=page_size, filters=filters)
            if not jobs:
                break
            for job in jobs:
                yield job
            if len(jobs) < page_size:
                break
            offset += page_size

    def _normalise_filters(
        self, filters: Mapping[str, Any]
    ) -> list[tuple[str, str]]:
        query: list[tuple[str, str]] = []
        for key, raw_value in filters.items():
            if raw_value is None:
                continue
            api_key = self._FILTER_ALIASES.get(key, key)
            if api_key not in self._FILTER_FIELDS:
                raise ValueError(f"Unsupported filter: {key}")
            query.append((api_key, self._serialise_filter_value(raw_value)))
        return query

    def _serialise_filter_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, datetime):
            iso = value.astimezone(UTC).isoformat()
            if iso.endswith("+00:00"):
                return iso[:-6] + "Z"
            return iso
        return str(value)

    def _build_url(self, path: str, query_pairs: Sequence[tuple[str, str]]) -> str:
        base = urljoin(self._base_url, path)
        query = urlencode(query_pairs)
        return f"{base}?{query}" if query else base

    def _perform_request(self, url: str) -> MutableMapping[str, Any]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        raw_payload = self._http_get(url, headers)
        if isinstance(raw_payload, Mapping):
            data = dict(raw_payload)
        else:
            if isinstance(raw_payload, bytes):
                text = raw_payload.decode("utf-8")
            else:
                text = str(raw_payload)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise JobsImportError("Jobs API returned invalid JSON") from exc
            if not isinstance(parsed, dict):
                raise JobsImportError("Jobs API response was not a JSON object")
            data = parsed
        return data

    def _default_http_get(self, url: str, headers: Mapping[str, str]) -> str:
        request = Request(url, headers=dict(headers), method="GET")
        try:
            with urlopen(request, timeout=self._timeout) as response:
                charset = response.headers.get_content_charset("utf-8")
                return response.read().decode(charset or "utf-8")
        except HTTPError as exc:  # pragma: no cover - network failure path
            raise JobsImportError(
                f"Jobs API request failed with status {exc.code}"
            ) from exc
        except URLError as exc:  # pragma: no cover - network failure path
            raise JobsImportError("Jobs API request failed") from exc
