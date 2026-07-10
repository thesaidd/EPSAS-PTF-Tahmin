from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.models import get_gpr_residual_ptf_service
from app.main import app
from ml.models.gpr_residual_ptf import GprResidualPtfService

ISTANBUL = ZoneInfo("Europe/Istanbul")


def residual_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=6, freq="h", tz=ISTANBUL)
    actual = np.array([100.0, 120.0, 140.0, 160.0, 180.0, 200.0])
    xgboost_prediction = np.array([95.0, 125.0, 135.0, 155.0, 190.0, 198.0])
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "xgboost_training_run_id": ["xgb-run"] * 6,
            "xgboost_model_version": ["xgboost_v1"] * 6,
            "xgboost_prediction": xgboost_prediction,
            "actual": actual,
            "xgboost_error": actual - xgboost_prediction,
            "xgboost_absolute_error": np.abs(actual - xgboost_prediction),
            "hour": [0, 1, 2, 3, 4, 5],
            "day_of_week": [3, 3, 3, 3, 3, 3],
            "month": [1, 1, 1, 1, 1, 1],
            "is_weekend": [False] * 6,
            "is_peak_hour": [False] * 6,
            "is_business_hour": [False] * 6,
            "ptf_lag_24": [90.0, 100.0, 120.0, 130.0, 150.0, 170.0],
            "ptf_lag_168": [80.0, 90.0, 110.0, 120.0, 140.0, 160.0],
            "ptf_24h_mean": [92.0, 102.0, 122.0, 132.0, 152.0, 172.0],
            "ptf_24h_std": [5.0] * 6,
            "ptf_7d_mean": [88.0, 98.0, 118.0, 128.0, 148.0, 168.0],
            "ptf_7d_std": [7.0] * 6,
            "residual": actual - xgboost_prediction,
        }
    )


def test_residual_calculation_contract() -> None:
    frame = residual_frame()

    assert frame["residual"].tolist() == pytest.approx(
        (frame["actual"] - frame["xgboost_prediction"]).tolist()
    )


def test_prepare_residual_features_converts_booleans() -> None:
    service = GprResidualPtfService()

    X, y, timestamps, xgb_predictions, actuals, fill_values = (
        service.prepare_residual_features(residual_frame())
    )

    assert len(X) == 6
    assert len(y) == 6
    assert len(timestamps) == 6
    assert len(xgb_predictions) == 6
    assert len(actuals) == 6
    assert set(X["is_weekend"].unique()) == {0.0}
    assert X.isna().sum().sum() == 0
    assert fill_values["ptf_24h_std"] == pytest.approx(5.0)


def test_recent_window_selection_uses_last_rows() -> None:
    service = GprResidualPtfService()
    selected, was_downsampled = service.select_recent_training_window(
        residual_frame(),
        max_train_rows=3,
    )

    assert was_downsampled is True
    assert selected["timestamp"].tolist() == residual_frame()["timestamp"].tail(3).tolist()


def test_risk_level_assignment_uses_interval_percentiles() -> None:
    service = GprResidualPtfService()
    frame = pd.DataFrame(
        {
            "interval_width_95": [10.0, 20.0, 30.0, 100.0],
        }
    )

    result = service.assign_risk_levels(frame)

    assert set(result["risk_level"]) == {"LOW", "MEDIUM", "HIGH"}
    assert result.loc[result["interval_width_95"] == 100.0, "risk_level"].item() == "HIGH"


def test_uncertainty_metrics_calculate_coverage_and_counts() -> None:
    service = GprResidualPtfService()
    frame = pd.DataFrame(
        {
            "actual": [100.0, 120.0, 200.0],
            "lower_bound_95": [90.0, 110.0, 150.0],
            "upper_bound_95": [110.0, 125.0, 190.0],
            "interval_width_95": [20.0, 15.0, 40.0],
            "risk_level": ["LOW", "LOW", "HIGH"],
        }
    )

    metrics = service.calculate_uncertainty_metrics(frame)

    assert metrics["interval_coverage_95"] == pytest.approx(2 / 3 * 100)
    assert metrics["mean_interval_width"] == pytest.approx(25.0)
    assert metrics["low_risk_count"] == 2
    assert metrics["high_risk_count"] == 1


class ConstantResidualModel:
    def predict(self, features: pd.DataFrame, return_std: bool = False):
        mean = np.full(len(features), 2.0)
        std = np.full(len(features), 5.0)
        return (mean, std) if return_std else mean


def test_xgboost_comparison_uses_same_prediction_frame_period() -> None:
    service = GprResidualPtfService()
    X, y, timestamps, xgb_predictions, actuals, _ = service.prepare_residual_features(
        residual_frame().tail(3)
    )

    metrics, predictions = service.evaluate_gpr_residual_model(
        ConstantResidualModel(),
        X,
        y,
        xgb_predictions,
        actuals,
        timestamps,
    )
    comparison = service.build_xgboost_comparison(predictions, metrics["mae"])

    assert comparison["comparison_window"] == "residual_test_period"
    assert comparison["xgboost_mae"] == pytest.approx(
        np.mean(np.abs(predictions["actual"] - predictions["xgboost_prediction"]))
    )


class FakePipeline:
    pass


def test_save_model_artifact_creates_joblib_file(tmp_path: Path) -> None:
    service = GprResidualPtfService(artifacts_root=tmp_path)

    artifact_path = service.save_model_artifact(
        FakePipeline(),
        model_version="gpr_residual_v1",
        gpr_run_id="gpr-run",
    )

    expected = tmp_path / "gpr_residual_v1" / "gpr-run" / "model.joblib"
    assert artifact_path == expected.as_posix()
    assert expected.exists()
    assert expected.stat().st_size > 0


class InMemoryGprService(GprResidualPtfService):
    def __init__(self, artifacts_root: Path) -> None:
        super().__init__(artifacts_root=artifacts_root)

    def get_latest_successful_xgboost_run(self) -> str | None:
        return "xgb-run"

    def load_residual_data(self, *args: object, **kwargs: object) -> pd.DataFrame:
        frame = residual_frame()
        frame["split"] = ["train", "train", "train", "test", "test", "test"]
        return frame

    def train_gpr_model(self, *args: object, **kwargs: object) -> ConstantResidualModel:
        return ConstantResidualModel()

    def compare_with_latest_baseline(self, gpr_mae: object) -> tuple[dict[str, object], str | None]:
        return (
            {
                "best_baseline_model": "naive_lag_24",
                "best_baseline_mae": 416.0,
                "gpr_corrected_mae": gpr_mae,
                "mae_improvement_pct": 10.0,
            },
            None,
        )

    def store_predictions(self, *args: object, **kwargs: object) -> int:
        return 3

    def store_metrics(self, *args: object, **kwargs: object) -> int:
        return 1

    def _log_to_mlflow(self, *args: object, **kwargs: object) -> str | None:
        return None


def test_run_residual_modeling_summary_shape(tmp_path: Path) -> None:
    summary = InMemoryGprService(tmp_path).run_residual_modeling(
        max_train_rows=2,
    )

    assert summary["gpr_run_id"]
    assert summary["xgboost_training_run_id"] == "xgb-run"
    assert summary["model_version"] == "gpr_residual_v1"
    assert summary["train_rows"] == 2
    assert summary["test_rows"] == 3
    assert summary["metrics"]["count"] == 3
    assert summary["uncertainty_metrics"]["interval_coverage_95"] is not None
    assert summary["artifact_path"]
    assert summary["errors"] == []


class FakeGprStatusService:
    def get_status(self) -> dict[str, object]:
        metric = {
            "mae": 1.0,
            "rmse": 2.0,
            "mape": 3.0,
            "smape": 4.0,
            "r2": 0.9,
            "count": 24,
            "mean_actual": 100.0,
            "mean_prediction": 99.0,
            "max_error": 5.0,
            "median_absolute_error": 1.0,
        }
        uncertainty = {
            "interval_coverage_95": 90.0,
            "mean_interval_width": 10.0,
            "median_interval_width": 9.0,
            "low_risk_count": 12,
            "medium_risk_count": 8,
            "high_risk_count": 4,
        }
        return {
            "total_prediction_rows": 24,
            "total_metric_rows": 1,
            "latest_gpr_run_id": "gpr-test-run",
            "latest_created_at": datetime(2026, 1, 2, tzinfo=ISTANBUL),
            "available_model_versions": ["gpr_residual_v1"],
            "latest_metrics": metric,
            "latest_uncertainty_metrics": uncertainty,
            "latest_xgboost_comparison": {"mae_improvement_pct": 5.0},
            "latest_baseline_comparison": {"mae_improvement_pct": 20.0},
            "latest_artifact_path": "artifacts/models/ptf/gpr_residual/model.joblib",
        }


def test_gpr_routes_are_registered_and_status_works() -> None:
    app.dependency_overrides[get_gpr_residual_ptf_service] = (
        lambda: FakeGprStatusService()
    )
    try:
        client = TestClient(app)
        response = client.get("/api/models/gpr-residual/ptf/status")
        paths = app.openapi()["paths"]
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["latest_gpr_run_id"] == "gpr-test-run"
    assert "get" in paths["/api/models/gpr-residual/ptf/status"]
    assert "post" in paths["/api/models/gpr-residual/ptf/train"]
