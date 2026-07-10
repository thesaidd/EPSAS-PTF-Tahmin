from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.models import get_xgboost_ptf_service
from app.main import app
from ml.models.xgboost_ptf import XGBoostPtfService, build_baseline_comparison

ISTANBUL = ZoneInfo("Europe/Istanbul")


def feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=3,
                freq="h",
                tz=ISTANBUL,
            ),
            "target_ptf": [100.0, 120.0, 140.0],
            "hour": [0, 1, 2],
            "is_weekend": [False, False, False],
            "is_peak_hour": [False, False, False],
            "season": ["winter", "winter", "winter"],
            "ptf_lag_24": [90.0, np.nan, 130.0],
            "ptf_lag_168": [80.0, 100.0, 120.0],
            "ptf_24h_mean": [95.0, 105.0, 125.0],
            "ptf_7d_mean": [85.0, 105.0, 125.0],
            "ptf_diff_1": [10.0, 20.0, 10.0],
            "ptf_pct_change_1": [0.1, 0.2, 0.1],
            "feature_version": ["v1", "v1", "v1"],
            "created_at": [datetime.now(tz=ISTANBUL)] * 3,
            "updated_at": [datetime.now(tz=ISTANBUL)] * 3,
        }
    )


def test_prepare_features_converts_booleans_and_encodes_season() -> None:
    X, y, timestamps, fill_values = XGBoostPtfService().prepare_features(
        feature_frame()
    )

    assert len(X) == 2
    assert len(y) == 2
    assert len(timestamps) == 2
    assert "season_winter" in X.columns
    assert "ptf_diff_1" not in X.columns
    assert "ptf_pct_change_1" not in X.columns
    assert "is_weekend" in X.columns
    assert set(X["is_weekend"].unique()) == {0.0}
    assert X.isna().sum().sum() == 0
    assert fill_values["ptf_lag_24"] == pytest.approx(110.0)


def test_prepare_features_reindexes_test_columns_to_training_columns() -> None:
    service = XGBoostPtfService()
    X_train, _, _, fill_values = service.prepare_features(feature_frame())
    test_frame = feature_frame().copy()
    test_frame["season"] = "summer"

    X_test, _, _, _ = service.prepare_features(
        test_frame,
        feature_columns=list(X_train.columns),
        fill_values=fill_values,
    )

    assert list(X_test.columns) == list(X_train.columns)
    assert "season_summer" not in X_test.columns
    assert "season_winter" in X_test.columns


class ConstantModel:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), 110.0)


def test_evaluate_model_uses_shared_metric_contract() -> None:
    service = XGBoostPtfService()
    X, y, timestamps, _ = service.prepare_features(feature_frame())

    metrics, predictions = service.evaluate_model(
        ConstantModel(),
        X,
        y,
        timestamps,
    )

    assert metrics["count"] == 2
    assert metrics["mae"] == pytest.approx(20.0)
    assert predictions["absolute_error"].tolist() == [10.0, 30.0]


def test_baseline_comparison_calculates_improvement() -> None:
    comparison = build_baseline_comparison(
        [
            {"model_name": "naive_lag_24", "mae": Decimal("416.32")},
            {"model_name": "seasonal_naive_lag_168", "mae": Decimal("442.31")},
        ],
        xgboost_mae=300.0,
    )

    assert comparison["best_baseline_model"] == "naive_lag_24"
    assert comparison["best_baseline_mae"] == pytest.approx(416.32)
    assert comparison["mae_improvement_pct"] == pytest.approx(
        (416.32 - 300.0) / 416.32 * 100
    )


class FakeNativeBooster:
    def __init__(self) -> None:
        self.saved_path: str | None = None

    def save_model(self, path: str) -> None:
        self.saved_path = path
        with open(path, "w", encoding="utf-8") as file:
            file.write('{"model":"native-booster"}')


class FakeSklearnWrapperWithBrokenSave:
    def __init__(self) -> None:
        self._Booster = FakeNativeBooster()

    def save_model(self, path: str) -> None:
        raise AttributeError(
            "`_estimator_type` undefined. Please use appropriate mixin "
            "to define estimator type."
        )


class FakeModelWithDirectSave:
    def __init__(self) -> None:
        self.saved_path: str | None = None

    def save_model(self, path: str) -> None:
        self.saved_path = path
        with open(path, "w", encoding="utf-8") as file:
            file.write('{"model":"direct-save"}')


def test_save_model_artifact_prefers_native_booster(tmp_path) -> None:
    model = FakeSklearnWrapperWithBrokenSave()
    service = XGBoostPtfService(artifacts_root=tmp_path)

    artifact_path = service.save_model_artifact(
        model,
        model_version="xgboost_v1",
        training_run_id="run-123",
    )

    expected_path = tmp_path / "xgboost_v1" / "run-123" / "model.json"
    assert artifact_path == expected_path.as_posix()
    assert expected_path.exists()
    assert model._Booster.saved_path == str(expected_path)


def test_save_model_artifact_falls_back_to_direct_save(tmp_path) -> None:
    model = FakeModelWithDirectSave()
    service = XGBoostPtfService(artifacts_root=tmp_path)

    artifact_path = service.save_model_artifact(
        model,
        model_version="xgboost_v1",
        training_run_id="run-456",
    )

    expected_path = tmp_path / "xgboost_v1" / "run-456" / "model.json"
    assert artifact_path == expected_path.as_posix()
    assert expected_path.exists()
    assert model.saved_path == str(expected_path)


class FakeXGBoostStatusService:
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
        return {
            "total_prediction_rows": 24,
            "total_metric_rows": 1,
            "latest_training_run_id": "xgb-test-run",
            "latest_created_at": datetime(2024, 1, 2, tzinfo=ISTANBUL),
            "available_model_versions": ["xgboost_v1"],
            "latest_metrics": metric,
            "latest_baseline_comparison": {
                "best_baseline_model": "naive_lag_24",
                "mae_improvement_pct": 10.0,
            },
        }


def test_xgboost_routes_are_registered_and_status_works() -> None:
    app.dependency_overrides[get_xgboost_ptf_service] = (
        lambda: FakeXGBoostStatusService()
    )
    try:
        client = TestClient(app)
        response = client.get("/api/models/xgboost/ptf/status")
        paths = app.openapi()["paths"]
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["latest_training_run_id"] == "xgb-test-run"
    assert "get" in paths["/api/models/xgboost/ptf/status"]
    assert "post" in paths["/api/models/xgboost/ptf/train"]
