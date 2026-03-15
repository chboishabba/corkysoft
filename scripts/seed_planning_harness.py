from __future__ import annotations

import argparse

from analytics.db.connection import connection_scope
from analytics.seed_harness import seed_mainland_jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed realistic mainland-Australia jobs and container requirements."
    )
    parser.add_argument("--db", default="routes.db", help="SQLite database path.")
    parser.add_argument("--count", type=int, default=10, help="Number of jobs to create.")
    parser.add_argument("--seed", type=int, default=20260314, help="Random seed.")
    parser.add_argument(
        "--baseline-containers",
        type=int,
        default=30,
        help="Baseline reusable container stock to seed.",
    )
    args = parser.parse_args()

    with connection_scope(args.db) as conn:
        rows = seed_mainland_jobs(
            conn,
            count=args.count,
            seed=args.seed,
            baseline_containers=args.baseline_containers,
        )

    print(f"Seeded {len(rows)} jobs into {args.db}")
    for row in rows:
        print(
            f"job#{row.job_id} {row.client}: {row.origin} -> {row.destination} | "
            f"{row.volume_m3:.1f}m3 | containers {row.allocated_containers}/{row.required_containers}"
        )


if __name__ == "__main__":
    main()

