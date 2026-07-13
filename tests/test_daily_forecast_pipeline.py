from datetime import date, datetime
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from app.api.pipelines import get_daily_forecast_pipeline_service
from app.main import app
from ml.pipelines.daily_forecast_pipeline import (
    DailyForecastPipelineService,
    STATUS_PARTIAL_SUCCESS,
    STATUS_SUCCESS,
)

ISTANBUL = ZoneInfo("Europe/Istanbul")


class FakePtfIngestionService:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def ingest_ptf_range(
        self,
        start_date: date,
        end_date: date,
        chunk_days: int = 3,
    ) -> dict[str, object]:
        if self.fail:
            raise RuntimeError("EPİAŞ unavailable")
        return {
            "chunks_processed": 1,
            "records_fetched": 24,
            "records_inserted_or_updated": 24,
            "missing_hours": [],
            "errors": [],
        }

    def get_status(self) -> dict[str, object]:
        return {"total_rows": 100}


class FakeFeatureService:
    def build_and_store_features(
        self,
        start_date: date,
        end_date: date,
        feature_version: str = "v1",
    ) -> dict[str, object]:
        return {
            "source_rows": 24,
            "feature_rows_built": 24,
            "feature_rows_inserted_or_updated": 24,
            "warnings": [],
            "errors": [],
        }


class FakeDayAheadService:
    def run_day_ahead_forecast(
        self,
        target_date: date | None = None,
        horizon_hours: int = 24,
        model_version: str = "day_ahead_v1",
    ) -> dict[str, object]:
        return {
            "forecast_run_id": "forecast-run",
            "target_date": (target_date or date(2026, 7, 11)).isoformat(),
            "rows_generated": 24,
            "selected_model": "xgboost",
            "warnings": [],
            "errors": [],
        }


class InMemoryDailyPipelineService(DailyForecastPipelineService):
    def __init__(self, ingestion_fails: bool = False) -> None:
        super().__init__(
            ptf_ingestion_service=FakePtfIngestionService(fail=ingestion_fails),
            feature_service=FakeFeatureService(),
            day_ahead_service=FakeDayAheadService(),
        )
        self.started_payloads: list[dict[str, object]] = []
        self.finished_payloads: list[dict[str, object]] = []

    def store_pipeline_run_start(self, **kwargs: object) -> None:
        self.started_payloads.append(kwargs)

    def update_pipeline_run_finish(self, *args: object, **kwargs: object) -> None:
        self.finished_payloads.append({"args": args, "kwargs": kwargs})


def test_pipeline_summary_shape_success() -> None:
    service = InMemoryDailyPipelineService()

    summary = service.run_pipeline(
        target_date=date(2026, 7, 11),
        ingest_start_date=date(2026, 7, 7),
        ingest_end_date=date(2026, 7, 10),
    )

    assert summary["pipeline_run_id"]
    assert summary["pipeline_name"] == "daily_forecast"
    assert summary["status"] == STATUS_SUCCESS
    assert summary["forecast_run_id"] == "forecast-run"
    assert summary["steps"]["ptf_ingestion"]["records_fetched"] == 24
    assert summary["steps"]["feature_build"]["feature_rows_built"] == 24
    assert summary["steps"]["day_ahead_forecast"]["rows_generated"] == 24
    assert summary["errors"] == []


def test_pipeline_partial_success_when_ingestion_fails_but_existing_data_available() -> None:
    service = InMemoryDailyPipelineService(ingestion_fails=True)

    summary = service.run_pipeline(
        target_date=date(2026, 7, 11),
        ingest_start_date=date(2026, 7, 7),
        ingest_end_date=date(2026, 7, 10),
    )

    assert summary["status"] == STATUS_PARTIAL_SUCCESS
    assert summary["steps"]["ptf_ingestion"]["status"] == "FAILED"
    assert summary["steps"]["feature_build"]["status"] == STATUS_SUCCESS
    assert summary["steps"]["day_ahead_forecast"]["status"] == STATUS_SUCCESS
    assert any("EPİAŞ unavailable" in error for error in summary["errors"])


def test_pipeline_run_storage_payload_shape() -> None:
    service = InMemoryDailyPipelineService()

    summary = service.run_pipeline(skip_ingestion=True, skip_feature_build=True)

    assert service.started_payloads
    assert service.finished_payloads
    assert service.started_payloads[0]["pipeline_run_id"] == summary["pipeline_run_id"]
    finish_args = service.finished_payloads[0]["args"]
    assert finish_args[0] == summary["pipeline_run_id"]
    assert finish_args[1] == STATUS_SUCCESS
    assert finish_args[3] == "forecast-run"


class FakePipelineApiService:
    def run_pipeline(self, **kwargs: object) -> dict[str, object]:
        return self._summary()

    def get_latest_pipeline_status(self) -> dict[str, object]:
        return self._summary()

    def get_pipeline_runs(self, limit: int = 20) -> list[dict[str, object]]:
        return [self._summary()]

    def _summary(self) -> dict[str, object]:
        return {
            "pipeline_run_id": "pipeline-run",
            "pipeline_name": "daily_forecast",
            "status": STATUS_SUCCESS,
            "started_at": datetime(2026, 7, 10, 13, tzinfo=ISTANBUL),
            "finished_at": datetime(2026, 7, 10, 13, 1, tzinfo=ISTANBUL),
            "target_date": date(2026, 7, 11),
            "ingest_start_date": date(2026, 7, 7),
            "ingest_end_date": date(2026, 7, 10),
            "forecast_run_id": "forecast-run",
            "steps": {
                "day_ahead_forecast": {
                    "status": STATUS_SUCCESS,
                    "forecast_run_id": "forecast-run",
                    "rows_generated": 24,
                }
            },
            "warnings": [],
            "errors": [],
        }


def test_daily_pipeline_routes_are_registered_and_status_works() -> None:
    app.dependency_overrides[get_daily_forecast_pipeline_service] = (
        lambda: FakePipelineApiService()
    )
    try:
        client = TestClient(app)
        run_response = client.post(
            "/api/pipelines/daily-forecast/run",
            json={
                "target_date": "2026-07-11",
                "ingest_start_date": "2026-07-07",
                "ingest_end_date": "2026-07-10",
                "skip_ingestion": True,
                "skip_feature_build": True,
            },
        )
        status_response = client.get("/api/pipelines/daily-forecast/status")
        runs_response = client.get("/api/pipelines/daily-forecast/runs")
        paths = app.openapi()["paths"]
    finally:
        app.dependency_overrides.clear()

    assert run_response.status_code == 200
    assert run_response.json()["forecast_run_id"] == "forecast-run"
    assert status_response.status_code == 200
    assert status_response.json()["pipeline_run_id"] == "pipeline-run"
    assert runs_response.status_code == 200
    assert len(runs_response.json()["runs"]) == 1
    assert "post" in paths["/api/pipelines/daily-forecast/run"]
    assert "get" in paths["/api/pipelines/daily-forecast/status"]
    assert "get" in paths["/api/pipelines/daily-forecast/runs"]


def test_pipeline_service_does_not_require_epias_credentials(
    monkeypatch,
) -> None:
    monkeypatch.delenv("EPIAS_USERNAME", raising=False)
    monkeypatch.delenv("EPIAS_PASSWORD", raising=False)

    service = InMemoryDailyPipelineService()

    assert service.run_pipeline(skip_ingestion=True)["status"] == STATUS_SUCCESS
