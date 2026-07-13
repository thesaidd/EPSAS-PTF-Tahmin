from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from app.api.monitoring import get_ptf_monitoring_service
from app.main import app
from ml.monitoring.ptf_monitoring import (
    PtfMonitoringService,
    STATUS_CRITICAL,
    STATUS_HEALTHY,
    STATUS_WARNING,
)

ISTANBUL = ZoneInfo("Europe/Istanbul")


def test_overall_status_aggregation() -> None:
    service = PtfMonitoringService()

    assert service.determine_overall_status(
        [{"status": STATUS_HEALTHY}, {"status": STATUS_HEALTHY}]
    ) == STATUS_HEALTHY
    assert service.determine_overall_status(
        [{"status": STATUS_HEALTHY}, {"status": STATUS_WARNING}]
    ) == STATUS_WARNING
    assert service.determine_overall_status(
        [{"status": STATUS_WARNING}, {"status": STATUS_CRITICAL}]
    ) == STATUS_CRITICAL


def test_data_quality_missing_hour_detection() -> None:
    section = PtfMonitoringService().evaluate_data_quality(
        expected_hours=24,
        actual_hours=22,
        missing_count=2,
        duplicate_count=0,
        negative_count=0,
        null_count=0,
    )

    assert section["status"] == STATUS_WARNING
    assert section["missing_hour_count"] == 2
    assert section["warnings"]


def test_forecast_health_with_24_rows_is_healthy() -> None:
    section = PtfMonitoringService().evaluate_forecast_health(
        forecast_run_id="forecast-run",
        target_date="2026-07-11",
        generated_at="2026-07-10T13:00:00+03:00",
        rows=24,
        distinct_horizon_hours=24,
        expected_horizon_hours=24,
    )

    assert section["status"] == STATUS_HEALTHY
    assert section["missing_horizon_hours"] == 0


def test_forecast_health_with_missing_rows_warns() -> None:
    section = PtfMonitoringService().evaluate_forecast_health(
        forecast_run_id="forecast-run",
        target_date="2026-07-11",
        generated_at="2026-07-10T13:00:00+03:00",
        rows=20,
        distinct_horizon_hours=20,
        expected_horizon_hours=24,
    )

    assert section["status"] == STATUS_WARNING
    assert section["missing_horizon_hours"] == 4


def test_uncertainty_coverage_thresholds() -> None:
    service = PtfMonitoringService()

    assert service.evaluate_uncertainty_status(90.0) == STATUS_HEALTHY
    assert service.evaluate_uncertainty_status(80.0) == STATUS_WARNING
    assert service.evaluate_uncertainty_status(99.0) == STATUS_WARNING
    assert service.evaluate_uncertainty_status(60.0) == STATUS_CRITICAL


def test_risk_summary_thresholds() -> None:
    service = PtfMonitoringService()
    healthy_rows = [
        {"forecast_ptf": 100.0, "interval_width_95": 20.0, "risk_level": "LOW"}
        for _ in range(20)
    ] + [
        {"forecast_ptf": 120.0, "interval_width_95": 40.0, "risk_level": "HIGH"}
        for _ in range(4)
    ]
    warning_rows = [
        {"forecast_ptf": 100.0, "interval_width_95": 20.0, "risk_level": "HIGH"}
        for _ in range(8)
    ]
    critical_rows = [
        {"forecast_ptf": 100.0, "interval_width_95": 20.0, "risk_level": "HIGH"}
        for _ in range(16)
    ]

    assert service.evaluate_risk_summary("run", healthy_rows)["status"] == STATUS_HEALTHY
    assert service.evaluate_risk_summary("run", warning_rows)["status"] == STATUS_WARNING
    assert service.evaluate_risk_summary("run", critical_rows)["status"] == STATUS_CRITICAL


class FakeMonitoringService:
    def build_snapshot(
        self,
        max_ptf_age_hours: int = 168,
        expected_forecast_horizon_hours: int = 24,
    ) -> dict[str, object]:
        return self._snapshot()

    def get_latest_snapshot(self) -> dict[str, object]:
        return self._snapshot()

    def get_compact_status(self) -> dict[str, object]:
        snapshot = self._snapshot()
        return {
            "status": snapshot["status"],
            "created_at": snapshot["created_at"],
            "warnings": [],
            "errors": [],
            "latest_pipeline_status": "SUCCESS",
            "latest_forecast_run_id": "forecast-run",
            "latest_data_timestamp": "2026-07-10T00:00:00+03:00",
            "latest_model_metrics": {"selected_model": "xgboost", "r2": 0.9},
        }

    def list_snapshots(self, limit: int = 20) -> list[dict[str, object]]:
        return [self._snapshot()]

    def _snapshot(self) -> dict[str, object]:
        section = {"status": STATUS_HEALTHY, "warnings": [], "errors": []}
        return {
            "snapshot_id": "snapshot-run",
            "status": STATUS_HEALTHY,
            "created_at": datetime(2026, 7, 10, 13, tzinfo=ISTANBUL),
            "data_freshness": {**section, "max_timestamp": "2026-07-10T00:00:00+03:00"},
            "data_quality": section,
            "pipeline_health": {**section, "latest_status": "SUCCESS"},
            "forecast_health": {**section, "latest_forecast_run_id": "forecast-run"},
            "model_quality": {**section, "latest_selected_model": "xgboost", "r2": 0.9},
            "uncertainty_quality": {**section, "interval_coverage_95": 90.0},
            "risk_summary": {**section, "high_risk_hours": 2},
            "warnings": [],
            "errors": [],
        }


def test_monitoring_routes_are_registered_and_snapshot_works() -> None:
    app.dependency_overrides[get_ptf_monitoring_service] = (
        lambda: FakeMonitoringService()
    )
    try:
        client = TestClient(app)
        snapshot_response = client.post(
            "/api/monitoring/ptf/snapshot",
            json={
                "max_ptf_age_hours": 168,
                "expected_forecast_horizon_hours": 24,
            },
        )
        latest_response = client.get("/api/monitoring/ptf/latest")
        status_response = client.get("/api/monitoring/ptf/status")
        snapshots_response = client.get("/api/monitoring/ptf/snapshots")
        paths = app.openapi()["paths"]
    finally:
        app.dependency_overrides.clear()

    assert snapshot_response.status_code == 200
    assert snapshot_response.json()["snapshot_id"] == "snapshot-run"
    assert latest_response.status_code == 200
    assert status_response.status_code == 200
    assert status_response.json()["latest_forecast_run_id"] == "forecast-run"
    assert snapshots_response.status_code == 200
    assert len(snapshots_response.json()["snapshots"]) == 1
    assert "post" in paths["/api/monitoring/ptf/snapshot"]
    assert "get" in paths["/api/monitoring/ptf/latest"]
    assert "get" in paths["/api/monitoring/ptf/status"]
    assert "get" in paths["/api/monitoring/ptf/snapshots"]


def test_snapshot_summary_shape_and_no_epias_credentials_required(monkeypatch) -> None:
    monkeypatch.delenv("EPIAS_USERNAME", raising=False)
    monkeypatch.delenv("EPIAS_PASSWORD", raising=False)

    snapshot = FakeMonitoringService().build_snapshot()

    assert snapshot["status"] == STATUS_HEALTHY
    assert "data_freshness" in snapshot
    assert "risk_summary" in snapshot
