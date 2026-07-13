import argparse
import json
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_API_URL = "http://localhost:8000"
DASHBOARD_URL = "http://localhost:8501"
SWAGGER_URL = "http://localhost:8000/docs"
MLFLOW_URL = "http://localhost:5000"


@dataclass(frozen=True)
class DemoStep:
    label: str
    method: str
    endpoint: str
    payload: dict[str, Any] | None = None


def build_demo_steps(
    run_forecast: bool = False,
    run_pipeline: bool = False,
    run_monitoring: bool = False,
) -> list[DemoStep]:
    steps = [
        DemoStep("API health", "GET", "/health"),
        DemoStep("System readiness", "GET", "/api/system/readiness"),
        DemoStep("EPİAŞ client health", "GET", "/api/epias/health"),
        DemoStep(
            "Day-ahead forecast status",
            "GET",
            "/api/forecasts/ptf/day-ahead/status",
        ),
        DemoStep(
            "Daily pipeline status",
            "GET",
            "/api/pipelines/daily-forecast/status",
        ),
        DemoStep("Monitoring status", "GET", "/api/monitoring/ptf/status"),
    ]
    if run_forecast:
        steps.append(
            DemoStep(
                "Generate day-ahead forecast",
                "POST",
                "/api/forecasts/ptf/day-ahead/generate",
                {"horizon_hours": 24, "model_version": "day_ahead_v1"},
            )
        )
    if run_pipeline:
        steps.append(
            DemoStep(
                "Run daily pipeline safely",
                "POST",
                "/api/pipelines/daily-forecast/run",
                {"skip_ingestion": True, "skip_feature_build": True},
            )
        )
    if run_monitoring:
        steps.append(
            DemoStep(
                "Create monitoring snapshot",
                "POST",
                "/api/monitoring/ptf/snapshot",
                {"max_ptf_age_hours": 168, "expected_forecast_horizon_hours": 24},
            )
        )
    return steps


def call_api(
    base_url: str,
    method: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
    timeout_seconds: int = 30,
) -> tuple[bool, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}{endpoint}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            return True, json.loads(body) if body else {}
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, str(exc)


def print_checklist() -> None:
    print("\nEPİAŞ PTF Forecasting MVP demo checklist")
    print("=" * 48)
    print("1. Start services: docker compose up -d --build")
    print("2. Open Swagger:   http://localhost:8000/docs")
    print("3. Open dashboard: http://localhost:8501")
    print("4. Open MLflow:    http://localhost:5000")
    print("5. Run this helper with --all for a safe local demo flow.")


def run_demo(args: argparse.Namespace) -> int:
    print_checklist()
    print("\nUseful URLs")
    print(f"- Dashboard: {DASHBOARD_URL}")
    print(f"- Swagger:   {SWAGGER_URL}")
    print(f"- MLflow:    {MLFLOW_URL}")

    run_forecast = args.all or args.run_forecast
    run_pipeline = args.all or args.run_pipeline
    run_monitoring = args.all or args.run_monitoring
    steps = build_demo_steps(run_forecast, run_pipeline, run_monitoring)

    print("\nEndpoint checks")
    failures = 0
    for step in steps:
        ok, result = call_api(args.api_url, step.method, step.endpoint, step.payload)
        marker = "OK" if ok else "FAIL"
        print(f"[{marker}] {step.label}: {step.method} {step.endpoint}")
        if ok:
            print(_compact_result(result))
        else:
            failures += 1
            print(f"  {result}")
    print("\nDemo helper finished.")
    return 1 if failures else 0


def _compact_result(result: Any) -> str:
    if not isinstance(result, dict):
        return f"  {result}"
    interesting = {
        key: result.get(key)
        for key in [
            "status",
            "api_healthy",
            "db_reachable",
            "latest_monitoring_status",
            "latest_forecast_run_id",
            "latest_pipeline_status",
            "forecast_run_id",
            "pipeline_run_id",
            "snapshot_id",
            "rows_generated",
            "total_rows",
            "total_runs",
        ]
        if key in result
    }
    return "  " + json.dumps(interesting or result, ensure_ascii=False, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe local MVP demo helper.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--run-forecast", action="store_true")
    parser.add_argument("--run-pipeline", action="store_true")
    parser.add_argument("--run-monitoring", action="store_true")
    parser.add_argument("--all", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_demo(parse_args()))
