from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.forecasts import get_day_ahead_ptf_service
from app.main import app
from ml.inference.day_ahead_ptf import (
    DayAheadPtfForecastService,
    assign_risk_levels,
    build_target_timestamps,
)

ISTANBUL = ZoneInfo("Europe/Istanbul")


def ptf_history(hours: int = 240) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-07-01 00:00",
        periods=hours,
        freq="h",
        tz=ISTANBUL,
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "ptf_tl": np.linspace(1000.0, 1400.0, hours),
        }
    )


def test_target_timestamps_cover_24_local_hours() -> None:
    timestamps = build_target_timestamps(date(2026, 7, 11), 24)

    assert len(timestamps) == 24
    assert timestamps[0].tzinfo is not None
    assert timestamps[0].hour == 0
    assert timestamps[-1].hour == 23
    assert timestamps[0].date() == date(2026, 7, 11)


def test_future_feature_frame_contains_calendar_and_lag_features() -> None:
    service = DayAheadPtfForecastService()

    features, warnings = service.build_future_feature_frame(
        target_date=date(2026, 7, 11),
        horizon_hours=24,
        history=ptf_history(),
    )

    assert len(features) == 24
    assert features["hour"].tolist() == list(range(24))
    assert {"ptf_lag_24", "ptf_24h_mean", "ptf_7d_mean", "season"}.issubset(
        features.columns
    )
    assert features["season"].unique().tolist() == ["summer"]
    assert any("does not recursively" in warning for warning in warnings)


def test_feature_alignment_preserves_order_and_fills_missing_columns() -> None:
    service = DayAheadPtfForecastService()
    raw_features, _ = service.build_future_feature_frame(
        target_date=date(2026, 7, 11),
        horizon_hours=2,
        history=ptf_history(),
    )

    aligned, warnings = service.align_feature_frame(
        raw_features,
        ["hour", "ptf_lag_24", "missing_training_column"],
    )

    assert aligned.columns.tolist() == [
        "hour",
        "ptf_lag_24",
        "missing_training_column",
    ]
    assert aligned["missing_training_column"].tolist() == [0.0, 0.0]
    assert any("missing_training_column" in warning for warning in warnings)


class FallbackUncertaintyService(DayAheadPtfForecastService):
    def get_historical_residual_std_fallback(self) -> float:
        return 42.0


def test_uncertainty_falls_back_when_gpr_artifact_missing() -> None:
    service = FallbackUncertaintyService()
    raw_features, _ = service.build_future_feature_frame(
        target_date=date(2026, 7, 11),
        horizon_hours=3,
        history=ptf_history(),
    )
    warnings: list[str] = []

    residual_mean, residual_std, gpr_run_id = service.predict_uncertainty(
        raw_features,
        np.array([100.0, 110.0, 120.0]),
        None,
        warnings,
    )

    assert residual_mean.tolist() == [0.0, 0.0, 0.0]
    assert residual_std.tolist() == [42.0, 42.0, 42.0]
    assert gpr_run_id is None


def test_risk_level_assignment_uses_interval_width_quantiles() -> None:
    frame = pd.DataFrame({"interval_width_95": [10.0, 20.0, 30.0, 100.0]})

    result = assign_risk_levels(frame)

    assert result["risk_level"].tolist() == ["LOW", "LOW", "MEDIUM", "HIGH"]


class FakeDayAheadForecastService:
    def run_day_ahead_forecast(
        self,
        target_date: date | None = None,
        horizon_hours: int = 24,
        model_version: str = "day_ahead_v1",
    ) -> dict[str, object]:
        return {
            "forecast_run_id": "forecast-run",
            "target_date": (target_date or date(2026, 7, 11)).isoformat(),
            "horizon_hours": horizon_hours,
            "selected_model": "xgboost",
            "xgboost_training_run_id": "xgb-run",
            "gpr_run_id": "gpr-run",
            "decision_run_id": "decision-run",
            "generation_method": "latest_history_calendar_lag_rolling_v1",
            "model_version": model_version,
            "rows_generated": horizon_hours,
            "min_timestamp": "2026-07-11T00:00:00+03:00",
            "max_timestamp": "2026-07-11T23:00:00+03:00",
            "warnings": [],
            "errors": [],
        }

    def get_latest_forecast(self) -> dict[str, object]:
        return {
            "latest_forecast_run_id": "forecast-run",
            "target_date": "2026-07-11",
            "generated_at": "2026-07-10T12:00:00+03:00",
            "rows": 1,
            "summary": {
                "mean_forecast": 100.0,
                "min_forecast": 100.0,
                "max_forecast": 100.0,
                "mean_interval_width": 20.0,
                "risk_level_counts": {"LOW": 1, "MEDIUM": 0, "HIGH": 0},
            },
            "forecasts": [
                {
                    "id": 1,
                    "forecast_run_id": "forecast-run",
                    "target_date": "2026-07-11",
                    "timestamp": "2026-07-11T00:00:00+03:00",
                    "horizon_hour": 1,
                    "selected_model": "xgboost",
                    "xgboost_prediction": 100.0,
                    "residual_mean": 0.0,
                    "residual_std": 5.0,
                    "forecast_ptf": 100.0,
                    "lower_bound_95": 90.2,
                    "upper_bound_95": 109.8,
                    "interval_width_95": 19.6,
                    "risk_level": "LOW",
                    "xgboost_training_run_id": "xgb-run",
                    "gpr_run_id": "gpr-run",
                    "decision_run_id": "decision-run",
                    "model_version": "day_ahead_v1",
                    "generation_method": "latest_history_calendar_lag_rolling_v1",
                    "warnings": [],
                    "generated_at": "2026-07-10T12:00:00+03:00",
                    "created_at": "2026-07-10T12:00:00+03:00",
                }
            ],
        }

    def get_status(self) -> dict[str, object]:
        return {
            "total_rows": 24,
            "total_runs": 1,
            "latest_forecast_run_id": "forecast-run",
            "latest_target_date": date(2026, 7, 11),
            "latest_generated_at": datetime(2026, 7, 10, 12, tzinfo=ISTANBUL),
            "available_model_versions": ["day_ahead_v1"],
        }


def test_day_ahead_forecast_routes_are_registered_and_generate_works() -> None:
    app.dependency_overrides[get_day_ahead_ptf_service] = (
        lambda: FakeDayAheadForecastService()
    )
    try:
        client = TestClient(app)
        response = client.post(
            "/api/forecasts/ptf/day-ahead/generate",
            json={
                "target_date": "2026-07-11",
                "horizon_hours": 24,
                "model_version": "day_ahead_v1",
            },
        )
        status_response = client.get("/api/forecasts/ptf/day-ahead/status")
        latest_response = client.get("/api/forecasts/ptf/day-ahead/latest")
        paths = app.openapi()["paths"]
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["rows_generated"] == 24
    assert response.json()["selected_model"] == "xgboost"
    assert status_response.status_code == 200
    assert status_response.json()["latest_forecast_run_id"] == "forecast-run"
    assert latest_response.status_code == 200
    assert latest_response.json()["summary"]["mean_forecast"] == 100.0
    assert "post" in paths["/api/forecasts/ptf/day-ahead/generate"]
    assert "get" in paths["/api/forecasts/ptf/day-ahead/latest"]
    assert "get" in paths["/api/forecasts/ptf/day-ahead/status"]


def test_day_ahead_service_does_not_require_epias_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EPIAS_USERNAME", raising=False)
    monkeypatch.delenv("EPIAS_PASSWORD", raising=False)

    service = DayAheadPtfForecastService()

    assert service is not None
