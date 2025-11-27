"""Minimal REST API surface for Corkysoft integrations."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

from analytics.db import connection_scope, fetch_driver_shifts


def _current_db_path() -> str:
    """Return the SQLite database path configured for the API."""

    return (
        os.environ.get("CORKYSOFT_DB")
        or os.environ.get("ROUTES_DB")
        or "routes.db"
    )


class JobType(BaseModel):
    """Descriptor for the type/category of a job."""

    code: str = Field(..., description="Short code representing the job type")
    text: Optional[str] = Field(
        default=None,
        description="Human readable text describing the job type",
    )


class BillingDetails(BaseModel):
    """Contact information for the billing entity."""

    name: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    address2: Optional[str] = None
    suburb: Optional[str] = None
    postcode: Optional[str] = None
    state: Optional[str] = None
    attention: Optional[str] = None
    code: Optional[str] = None


class ServiceDescriptor(BaseModel):
    """Identifier for the underlying service or product."""

    code: Optional[str] = None
    text: Optional[str] = None


class JobResponse(BaseModel):
    """Response payload for ``GET /jobs/{jobId}``."""

    id: str = Field(..., description="Primary identifier for the job")
    externalId: Optional[str] = Field(
        default=None,
        description="External system identifier for the job",
    )
    created: Optional[str] = Field(
        default=None,
        description="ISO8601 timestamp when the job was created",
    )
    lastUpdated: Optional[str] = Field(
        default=None,
        description="ISO8601 timestamp when the job was last updated",
    )
    lastUpdatedBy: Optional[str] = Field(
        default=None,
        description="User identifier responsible for the latest update",
    )
    type: JobType = Field(
        default_factory=lambda: JobType(code="R", text="Move"),
        description="Categorisation metadata for the job",
    )
    billing: Optional[BillingDetails] = Field(
        default=None,
        description="Billing contact details for the job",
    )
    service: Optional[ServiceDescriptor] = Field(
        default=None,
        description="Service information associated with the job",
    )
    origin: Optional[str] = Field(
        default=None,
        description="Free form origin description captured for the job",
    )
    destination: Optional[str] = Field(
        default=None,
        description="Free form destination description captured for the job",
    )
    revenue: Optional[float] = Field(
        default=None,
        description="Recorded revenue attributable to the job",
    )
    revenueTotal: Optional[float] = Field(
        default=None,
        description="Total revenue including adjustments",
    )
    volume: Optional[float] = Field(
        default=None,
        description="Volume captured for the job in native units",
    )
    volumeM3: Optional[float] = Field(
        default=None,
        description="Volume captured for the job expressed in cubic metres",
    )
    distanceKm: Optional[float] = Field(
        default=None,
        description="Distance for the job measured in kilometres",
    )
    finalCost: Optional[float] = Field(
        default=None,
        description="Final cost assigned to the job",
    )


class DriverShiftResponse(BaseModel):
    """Representation of a driver shift entry."""

    id: int = Field(..., description="Primary identifier for the shift")
    shiftDate: str = Field(..., description="Shift date in ISO format")
    truckId: Optional[str] = Field(
        default=None, description="Truck identifier attached to the shift"
    )
    truckName: Optional[str] = Field(
        default=None, description="Friendly truck name if available"
    )
    workerId: Optional[int] = Field(
        default=None, description="Worker identifier attached to the shift"
    )
    workerName: Optional[str] = Field(
        default=None, description="Driver or worker name for the shift"
    )
    ticketNumbers: Optional[str] = Field(
        default=None, description="Ticket numbers or references logged"
    )
    shiftStart: Optional[str] = Field(
        default=None, description="Recorded start time for the shift"
    )
    shiftEnd: Optional[str] = Field(
        default=None, description="Recorded finish time for the shift"
    )
    hours: Optional[float] = Field(
        default=None, description="Duration of the shift in hours"
    )
    hourlyRate: Optional[float] = Field(
        default=None, description="Rate applied to the shift"
    )
    costTotal: Optional[float] = Field(
        default=None, description="Total cost recorded for the shift"
    )
    source: Optional[str] = Field(
        default=None, description="Origin of the shift record (e.g. sheet tab)"
    )
    importedAt: Optional[str] = Field(
        default=None, description="Timestamp when the shift was last imported"
    )


class MovewareImportRequest(BaseModel):
    """Request payload for the MoveWare importer endpoint."""

    records: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Collection of raw MoveWare job dictionaries",
    )
    dry_run: bool = Field(
        default=False,
        description="When true, only validate the payload without persisting",
    )


class ImportSummary(BaseModel):
    """Summary of a MoveWare import invocation."""

    resource: str = Field(
        ..., description="Specific MoveWare resource targeted by the import"
    )
    imported: int = Field(
        ..., description="Number of records supplied to the importer"
    )
    dry_run: bool = Field(
        ..., description="Whether the invocation was executed in dry-run mode"
    )


app = FastAPI(title="Corkysoft API", version="0.1.0")


def _optional_column(row: Any, column: str) -> Optional[Any]:
    """Return ``row[column]`` if the column exists, otherwise ``None``."""

    if hasattr(row, "keys") and column in row.keys():
        return row[column]
    try:
        return row[column]
    except (KeyError, TypeError, IndexError):
        return None


def _build_job_response(row: Any) -> JobResponse:
    """Transform a SQLite row into a :class:`JobResponse`."""

    job_type_code = _optional_column(row, "job_type_code") or "R"
    job_type_text = _optional_column(row, "job_type_text")
    if not job_type_text and job_type_code == "R":
        job_type_text = "Move"

    billing = BillingDetails(
        name=_optional_column(row, "billing_name")
        or _optional_column(row, "client"),
        email=_optional_column(row, "billing_email"),
        address=_optional_column(row, "billing_address"),
        address2=_optional_column(row, "billing_address2"),
        suburb=_optional_column(row, "billing_suburb"),
        postcode=_optional_column(row, "billing_postcode"),
        state=_optional_column(row, "billing_state"),
        attention=_optional_column(row, "billing_attention"),
        code=_optional_column(row, "billing_code"),
    )
    billing_payload = billing if billing.model_dump(exclude_none=True) else None

    service = ServiceDescriptor(
        code=_optional_column(row, "service_code"),
        text=_optional_column(row, "service_text"),
    )
    service_payload = service if service.model_dump(exclude_none=True) else None

    return JobResponse(
        id=str(_optional_column(row, "id")),
        externalId=_optional_column(row, "external_id"),
        created=_optional_column(row, "created_at")
        or _optional_column(row, "job_date"),
        lastUpdated=_optional_column(row, "updated_at"),
        lastUpdatedBy=_optional_column(row, "last_updated_by"),
        type=JobType(code=job_type_code, text=job_type_text),
        billing=billing_payload,
        service=service_payload,
        origin=_optional_column(row, "origin"),
        destination=_optional_column(row, "destination"),
        revenue=_optional_column(row, "revenue"),
        revenueTotal=_optional_column(row, "revenue_total"),
        volume=_optional_column(row, "volume"),
        volumeM3=_optional_column(row, "volume_m3"),
        distanceKm=_optional_column(row, "distance_km"),
        finalCost=_optional_column(row, "final_cost"),
    )


@app.post(
    "/importers/moveware/{resource}",
    response_model=ImportSummary,
    summary="Import MoveWare data",
)
def import_moveware_resource(
    resource: str = Path(..., description="MoveWare resource identifier"),
    payload: MovewareImportRequest = ...,
) -> ImportSummary:
    """Return a summary for an incoming MoveWare import request."""

    return ImportSummary(
        resource=resource,
        imported=len(payload.records),
        dry_run=payload.dry_run,
    )


@app.get(
    "/jobs/{jobId}",
    response_model=JobResponse,
    summary="Fetch a job by its identifier",
)
def get_job(jobId: str = Path(..., description="Unique job identifier")) -> JobResponse:
    """Return a job from the ``jobs`` table."""

    with connection_scope(_current_db_path()) as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (jobId,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    job_id = _optional_column(row, "id")
    if job_id is None:
        raise HTTPException(status_code=500, detail="Job row missing identifier")
    return _build_job_response(row)


@app.get(
    "/driver-shifts",
    response_model=List[DriverShiftResponse],
    summary="List driver shifts with optional filters",
)
def list_driver_shifts(
    start_date: Optional[str] = Query(
        default=None, description="Earliest shift date (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = Query(
        default=None, description="Latest shift date (YYYY-MM-DD)"
    ),
    workers: Optional[List[str]] = Query(
        default=None, description="Filter results to specific worker names"
    ),
    trucks: Optional[List[str]] = Query(
        default=None, description="Filter results to specific truck identifiers"
    ),
) -> List[DriverShiftResponse]:
    """Return driver shifts stored in the ``driver_shifts`` table."""

    with connection_scope(_current_db_path()) as conn:
        rows = fetch_driver_shifts(
            conn,
            start_date=start_date,
            end_date=end_date,
            worker_names=workers,
            truck_ids=trucks,
        )

    responses: List[DriverShiftResponse] = []
    for row in rows:
        responses.append(
            DriverShiftResponse(
                id=int(row["id"]),
                shiftDate=row["shift_date"],
                truckId=row["truck_id"],
                truckName=row["truck_name"],
                workerId=row["worker_id"],
                workerName=row["worker_name"],
                ticketNumbers=row["ticket_numbers"],
                shiftStart=row["shift_start"],
                shiftEnd=row["shift_end"],
                hours=row["hours"],
                hourlyRate=row["hourly_rate"],
                costTotal=row["cost_total"],
                source=row["source"],
                importedAt=row["imported_at"],
            )
        )
    return responses


__all__ = [
    "app",
    "JobResponse",
    "DriverShiftResponse",
    "MovewareImportRequest",
    "ImportSummary",
]
