"""Load generator for the inference API.

Fires N requests at a fixed concurrency and reports the latency distribution
plus the **peak queue depth** observed during the run. Queue depth is the point
of the exercise: in-flight count saturates at ``max_concurrency`` and stops
being informative the moment the service is busy, whereas queue depth keeps
growing and shows how far behind the service is falling.

Usage::

    python scripts/load_test.py --n 300 --concurrency 16 --model deep
    python scripts/load_test.py --n 500 --concurrency 32 --rows 50

The service must already be running::

    uvicorn api.main:app --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import sys
import time
from pathlib import Path

import httpx

# Allow `python scripts/load_test.py` without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from titanic.config import Paths  # noqa: E402
from titanic.data import load_csv  # noqa: E402

# httpx logs one INFO line per request, which would bury the report under 300
# lines of noise. The report itself is the output of this script.
logging.getLogger("httpx").setLevel(logging.WARNING)

#: How often to sample /stats for the peak queue depth, in seconds. Frequent
#: enough to catch a transient spike, cheap enough not to perturb the run.
POLL_INTERVAL = 0.05


def build_payload(n_rows: int, model: str | None, threshold: float) -> dict:
    """Build a /predict body from the committed sample dataset.

    Real rows are used rather than synthetic ones so preprocessing does the
    same work it would in production -- title extraction, deck parsing and
    imputation all behave differently on tidy fake data.

    Args:
        n_rows: Passengers per request.
        model: Model name, or ``None`` for the service default.
        threshold: Decision threshold.

    Returns:
        A JSON-serialisable request body.
    """
    frame = load_csv(Paths().sample_csv)
    # Cycle rather than sample: every run sends identical payloads, so two
    # runs are comparable.
    selected = frame.iloc[[i % len(frame) for i in range(n_rows)]]

    passengers = []
    for row in selected.to_dict(orient="records"):
        passenger = {
            key: row[key]
            for key in ("Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")
            if key in row and row[key] == row[key]  # NaN is the only value != itself
        }
        passengers.append(passenger)

    body: dict = {"passengers": passengers, "threshold": threshold}
    if model:
        body["model"] = model
    return body


async def fire_one(
    client: httpx.AsyncClient, url: str, payload: dict, semaphore: asyncio.Semaphore
) -> tuple[float, int]:
    """Send one request and time it.

    Args:
        client: Shared HTTP client.
        url: Target URL.
        payload: Request body.
        semaphore: Limits how many requests are in flight from this client.

    Returns:
        ``(latency_ms, status_code)``; status 0 means the request never
        completed (connection error or timeout).
    """
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await client.post(url, json=payload, timeout=30.0)
            status = response.status_code
        except httpx.HTTPError:
            status = 0
        return (time.perf_counter() - started) * 1000, status


async def poll_queue_depth(
    client: httpx.AsyncClient, base_url: str, stop: asyncio.Event
) -> dict[str, int]:
    """Sample /stats until told to stop, tracking the peak queue depth.

    The gauge itself is nearly always zero by the time a run finishes, so the
    peak has to be captured while load is actually being applied.

    Args:
        client: Shared HTTP client.
        base_url: Service base URL.
        stop: Event signalling the end of the run.

    Returns:
        ``{"peak_queue_depth": ..., "peak_inflight": ..., "samples": ...}``.
    """
    peak_depth = 0
    peak_inflight = 0
    samples = 0

    while not stop.is_set():
        try:
            queue = (await client.get(f"{base_url}/stats", timeout=5.0)).json()["queue"]
            peak_depth = max(peak_depth, queue.get("depth", 0))
            peak_inflight = max(peak_inflight, queue.get("inflight", 0))
            samples += 1
        except (httpx.HTTPError, KeyError, ValueError):
            # Polling is diagnostic, not load: a failed sample must never
            # abort the run or be counted as a request error.
            pass
        await asyncio.sleep(POLL_INTERVAL)

    return {
        "peak_queue_depth": peak_depth,
        "peak_inflight": peak_inflight,
        "samples": samples,
    }


async def run_load_test(args: argparse.Namespace) -> dict:
    """Execute the load test and return its summary.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A dict of results suitable for printing or embedding in the app.
    """
    base_url = args.url.rstrip("/")
    payload = build_payload(args.rows, args.model, args.threshold)
    semaphore = asyncio.Semaphore(args.concurrency)
    stop = asyncio.Event()

    async with httpx.AsyncClient() as client:
        try:
            health = await client.get(f"{base_url}/health", timeout=5.0)
            health.raise_for_status()
        except httpx.HTTPError as exc:
            raise SystemExit(
                f"Cannot reach the API at {base_url} ({exc}).\n"
                "Start it first:  uvicorn api.main:app --port 8000"
            ) from exc

        poller = asyncio.create_task(poll_queue_depth(client, base_url, stop))

        started = time.perf_counter()
        results = await asyncio.gather(
            *(fire_one(client, f"{base_url}/predict", payload, semaphore) for _ in range(args.n))
        )
        wall_seconds = time.perf_counter() - started

        stop.set()
        queue_stats = await poller

        final_stats = (await client.get(f"{base_url}/stats", timeout=5.0)).json()

    latencies = sorted(latency for latency, _ in results)
    statuses = [status for _, status in results]
    ok = sum(1 for status in statuses if status == 200)
    rejected = sum(1 for status in statuses if status == 503)
    failed = len(statuses) - ok - rejected

    def percentile(fraction: float) -> float:
        """Nearest-rank percentile over the measured latencies."""
        if not latencies:
            return 0.0
        return latencies[min(int(fraction * len(latencies)), len(latencies) - 1)]

    return {
        "n": args.n,
        "concurrency": args.concurrency,
        "model": args.model or final_stats.get("models", {}) and args.model,
        "rows_per_request": args.rows,
        "wall_seconds": round(wall_seconds, 2),
        "rps": round(args.n / wall_seconds, 1) if wall_seconds else 0.0,
        "ok": ok,
        "rejected_503": rejected,
        "failed": failed,
        "p50_ms": round(percentile(0.50), 2),
        "p95_ms": round(percentile(0.95), 2),
        "p99_ms": round(percentile(0.99), 2),
        "max_ms": round(latencies[-1], 2) if latencies else 0.0,
        "mean_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        **queue_stats,
        "server_error_rate": final_stats["requests"]["error_rate"],
        "server_stage_latency": final_stats["latency_ms"]["per_stage"],
    }


def format_report(summary: dict) -> str:
    """Render the summary as the block the README quotes.

    Args:
        summary: Output of :func:`run_load_test`.

    Returns:
        A printable multi-line report.
    """
    lines = [
        f"load_test: n={summary['n']} concurrency={summary['concurrency']} "
        f"model={summary['model'] or 'default'} rows/req={summary['rows_per_request']}",
        f"  p50={summary['p50_ms']} ms  p95={summary['p95_ms']} ms  "
        f"p99={summary['p99_ms']} ms  max={summary['max_ms']} ms",
        f"  ok={summary['ok']}  rejected(503)={summary['rejected_503']}  "
        f"failed={summary['failed']}  rps={summary['rps']}",
        f"  peak_queue_depth={summary['peak_queue_depth']}  "
        f"peak_inflight={summary['peak_inflight']}  "
        f"({summary['samples']} stats samples over {summary['wall_seconds']}s)",
    ]

    stages = summary.get("server_stage_latency") or {}
    if stages:
        breakdown = "  ".join(f"{stage}={values['p95']}ms" for stage, values in stages.items())
        lines.append(f"  server p95 by stage: {breakdown}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command line.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python scripts/load_test.py",
        description="Drive load at the inference API and report latency and queue depth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--n", type=int, default=300, help="total requests")
    parser.add_argument("--concurrency", type=int, default=16, help="requests in flight")
    parser.add_argument("--rows", type=int, default=1, help="passengers per request")
    parser.add_argument("--model", default=None, help="model name (default: service default)")
    parser.add_argument("--threshold", type=float, default=0.5, help="decision threshold")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the load test and print its report.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Exit code: 0 if every request succeeded or was cleanly rejected.
    """
    args = parse_args(argv)
    summary = asyncio.run(run_load_test(args))
    print(format_report(summary))
    # A 503 is a correct response under back-pressure, so only genuine
    # failures (connection errors, 5xx other than queue rejection) fail the run.
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
