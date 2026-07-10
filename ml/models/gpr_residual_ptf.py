import json
import logging
import uuid
from collections.abc import Callable
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal, engine
from ml.evaluation.metrics import calculate_regression_metrics
from ml.models.xgboost_ptf import build_baseline_comparison

logger = logging.getLogger(__name__)

ISTANBUL_TIMEZONE = ZoneInfo("Europe/Istanbul")
DEFAULT_MODEL_VERSION = "gpr_residual_v1"
DEFAULT_RESIDUAL_TRAIN_START = date(2024, 1, 1)
DEFAULT_RESIDUAL_TRAIN_END = date(2025, 12, 31)
DEFAULT_RESIDUAL_TEST_START = date(2026, 1, 1)
DEFAULT_RESIDUAL_TEST_END = date(2026, 7, 9)
DEFAULT_MAX_TRAIN_ROWS = 3000
MLFLOW_EXPERIMENT_NAME = "ptf_gpr_residual_forecasting"
GPR_FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak_hour",
    "is_business_hour",
    "ptf_lag_24",
    "ptf_lag_168",
    "ptf_24h_mean",
    "ptf_24h_std",
    "ptf_7d_mean",
    "ptf_7d_std",
    "xgboost_prediction",
]
METRIC_NAMES = (
    "mae",
    "rmse",
    "mape",
    "smape",
    "r2",
    "count",
    "mean_actual",
    "mean_prediction",
    "max_error",
    "median_absolute_error",
)
UNCERTAINTY_METRIC_NAMES = (
    "interval_coverage_95",
    "mean_interval_width",
    "median_interval_width",
    "low_risk_count",
    "medium_risk_count",
    "high_risk_count",
)
DEFAULT_GPR_PARAMS: dict[str, Any] = {
    "constant_value": 1.0,
    "rbf_length_scale": 1.0,
    "white_noise_level": 1.0,
    "normalize_y": True,
    "random_state": 42,
    # Optimizer restarts make exact GPR materially slower. The kernel is still
    # explicit and tunable, but MVP training remains practical by default.
    "optimizer": None,
    "n_restarts_optimizer": 0,
}


class GprResidualPtfService:
    def __init__(
        self,
        session_factory: Callable[[], Session] = SessionLocal,
        mlflow_tracking_uri: str | None = None,
        artifacts_root: Path | str = "artifacts/models/ptf/gpr_residual",
    ) -> None:
        self.session_factory = session_factory
        self.mlflow_tracking_uri = mlflow_tracking_uri or settings.mlflow_tracking_uri
        self.artifacts_root = Path(artifacts_root)

    def get_latest_successful_xgboost_run(self) -> str | None:
        with self.session_factory() as session:
            return session.scalar(
                text(
                    """
                    SELECT training_run_id
                    FROM xgboost_metrics
                    WHERE artifact_path IS NOT NULL
                      AND artifact_path <> ''
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                )
            )

    def load_residual_data(
        self,
        xgboost_training_run_id: str,
        residual_train_start: date | datetime,
        residual_train_end: date | datetime,
        residual_test_start: date | datetime,
        residual_test_end: date | datetime,
    ) -> pd.DataFrame:
        train_start = _normalize_boundary(residual_train_start, is_end=False)
        train_end = _normalize_boundary(residual_train_end, is_end=True)
        test_start = _normalize_boundary(residual_test_start, is_end=False)
        test_end = _normalize_boundary(residual_test_end, is_end=True)
        if train_end < train_start:
            raise ValueError("residual_train_end must be on or after residual_train_start")
        if test_end < test_start:
            raise ValueError("residual_test_end must be on or after residual_test_start")

        query_start = min(train_start, test_start)
        query_end = max(train_end, test_end)
        query = text(
            """
            SELECT
                p."timestamp",
                p.training_run_id AS xgboost_training_run_id,
                p.model_version AS xgboost_model_version,
                p.prediction AS xgboost_prediction,
                p.actual,
                p.error AS xgboost_error,
                p.absolute_error AS xgboost_absolute_error,
                f.hour,
                f.day_of_week,
                f.month,
                f.is_weekend,
                f.is_peak_hour,
                f.is_business_hour,
                f.ptf_lag_24,
                f.ptf_lag_168,
                f.ptf_24h_mean,
                f.ptf_24h_std,
                f.ptf_7d_mean,
                f.ptf_7d_std
            FROM xgboost_predictions AS p
            JOIN features_ptf_hourly AS f
              ON f."timestamp" = p."timestamp"
            WHERE p.training_run_id = :training_run_id
              AND p."timestamp" >= :query_start
              AND p."timestamp" <= :query_end
            ORDER BY p."timestamp"
            """
        )
        with engine.connect() as connection:
            dataframe = pd.read_sql_query(
                query,
                connection,
                params={
                    "training_run_id": xgboost_training_run_id,
                    "query_start": query_start,
                    "query_end": query_end,
                },
            )

        if dataframe.empty:
            return dataframe
        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=True)
        numeric_columns = [
            "xgboost_prediction",
            "actual",
            "xgboost_error",
            "xgboost_absolute_error",
            "hour",
            "day_of_week",
            "month",
            "ptf_lag_24",
            "ptf_lag_168",
            "ptf_24h_mean",
            "ptf_24h_std",
            "ptf_7d_mean",
            "ptf_7d_std",
        ]
        for column in numeric_columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
        for column in ["is_weekend", "is_peak_hour", "is_business_hour"]:
            dataframe[column] = dataframe[column].astype(bool)
        dataframe["residual"] = dataframe["actual"] - dataframe["xgboost_prediction"]
        dataframe["split"] = np.where(
            (dataframe["timestamp"] >= pd.Timestamp(test_start))
            & (dataframe["timestamp"] <= pd.Timestamp(test_end)),
            "test",
            np.where(
                (dataframe["timestamp"] >= pd.Timestamp(train_start))
                & (dataframe["timestamp"] <= pd.Timestamp(train_end)),
                "train",
                "unused",
            ),
        )
        return dataframe.sort_values("timestamp").reset_index(drop=True)

    def prepare_residual_features(
        self,
        dataframe: pd.DataFrame,
        feature_columns: list[str] | None = None,
        fill_values: dict[str, float] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, dict[str, float]]:
        required = {"timestamp", "residual", "actual", "xgboost_prediction"}
        missing = required.difference(dataframe.columns)
        if missing:
            raise ValueError(
                f"Missing residual modeling columns: {', '.join(sorted(missing))}"
            )

        frame = dataframe.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        for column in ["residual", "actual", "xgboost_prediction"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.sort_values("timestamp").dropna(
            subset=["timestamp", "residual", "actual", "xgboost_prediction"]
        )

        feature_frame = pd.DataFrame(index=frame.index)
        source_columns = feature_columns or GPR_FEATURE_COLUMNS
        for column in source_columns:
            if column not in frame.columns:
                feature_frame[column] = np.nan
                continue
            series = frame[column]
            if pd.api.types.is_bool_dtype(series):
                feature_frame[column] = series.astype(int)
            else:
                feature_frame[column] = pd.to_numeric(series, errors="coerce")

        if feature_columns is None:
            feature_columns = list(feature_frame.columns)
        feature_frame = feature_frame.reindex(columns=feature_columns)
        if fill_values is None:
            fill_values = {
                column: _finite_median(feature_frame[column])
                for column in feature_columns
            }
        feature_frame = feature_frame.fillna(fill_values).fillna(0.0).astype(float)

        target = frame["residual"].astype(float)
        timestamps = frame["timestamp"]
        actuals = frame["actual"].astype(float)
        xgboost_predictions = frame["xgboost_prediction"].astype(float)
        finite_rows = (
            np.isfinite(feature_frame.to_numpy(dtype=float)).all(axis=1)
            & np.isfinite(target.to_numpy(dtype=float))
            & np.isfinite(actuals.to_numpy(dtype=float))
            & np.isfinite(xgboost_predictions.to_numpy(dtype=float))
        )
        return (
            feature_frame.loc[finite_rows].reset_index(drop=True),
            target.loc[finite_rows].reset_index(drop=True),
            timestamps.loc[finite_rows].reset_index(drop=True),
            xgboost_predictions.loc[finite_rows].reset_index(drop=True),
            actuals.loc[finite_rows].reset_index(drop=True),
            fill_values,
        )

    def select_recent_training_window(
        self,
        dataframe: pd.DataFrame,
        max_train_rows: int,
    ) -> tuple[pd.DataFrame, bool]:
        if max_train_rows <= 0:
            raise ValueError("max_train_rows must be positive")
        sorted_frame = dataframe.sort_values("timestamp").reset_index(drop=True)
        if len(sorted_frame) <= max_train_rows:
            return sorted_frame, False
        return sorted_frame.tail(max_train_rows).reset_index(drop=True), True

    def train_gpr_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: dict[str, Any] | None = None,
    ) -> Any:
        if X_train.empty or y_train.empty:
            raise ValueError("Residual training data is empty")

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        model_params = {**DEFAULT_GPR_PARAMS, **(params or {})}
        kernel = (
            ConstantKernel(float(model_params["constant_value"]), (1e-3, 1e3))
            * RBF(length_scale=float(model_params["rbf_length_scale"]))
            + WhiteKernel(noise_level=float(model_params["white_noise_level"]))
        )
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=bool(model_params["normalize_y"]),
            random_state=int(model_params["random_state"]),
            optimizer=model_params["optimizer"],
            n_restarts_optimizer=int(model_params["n_restarts_optimizer"]),
        )
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("gpr", gpr),
            ]
        )
        model.fit(X_train, y_train)
        return model

    def evaluate_gpr_residual_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        xgboost_predictions: pd.Series,
        actuals: pd.Series,
        timestamps: pd.Series,
    ) -> tuple[dict[str, float | int | None], pd.DataFrame]:
        residual_mean, residual_std = model.predict(X_test, return_std=True)
        residual_std = np.maximum(np.asarray(residual_std, dtype=float), 0.0)
        final_prediction = xgboost_predictions.to_numpy(dtype=float) + residual_mean
        prediction_frame = pd.DataFrame(
            {
                "timestamp": timestamps.reset_index(drop=True),
                "xgboost_prediction": xgboost_predictions.reset_index(drop=True),
                "actual": actuals.reset_index(drop=True),
                "residual_mean": residual_mean,
                "residual_std": residual_std,
                "final_prediction": final_prediction,
            }
        )
        prediction_frame["lower_bound_95"] = (
            prediction_frame["final_prediction"] - 1.96 * prediction_frame["residual_std"]
        )
        prediction_frame["upper_bound_95"] = (
            prediction_frame["final_prediction"] + 1.96 * prediction_frame["residual_std"]
        )
        prediction_frame["interval_width_95"] = (
            prediction_frame["upper_bound_95"] - prediction_frame["lower_bound_95"]
        )
        prediction_frame = self.assign_risk_levels(prediction_frame)
        prediction_frame["error"] = (
            prediction_frame["actual"] - prediction_frame["final_prediction"]
        )
        prediction_frame["absolute_error"] = prediction_frame["error"].abs()
        denominator = prediction_frame["actual"].abs().replace(0, np.nan)
        prediction_frame["percentage_error"] = (
            prediction_frame["absolute_error"] / denominator * 100
        )
        metrics = calculate_regression_metrics(
            prediction_frame["actual"],
            prediction_frame["final_prediction"],
        )
        return metrics, prediction_frame

    def assign_risk_levels(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = dataframe.copy()
        if frame.empty:
            frame["risk_level"] = pd.Series(dtype="object")
            return frame
        q50 = frame["interval_width_95"].quantile(0.50)
        q85 = frame["interval_width_95"].quantile(0.85)
        frame["risk_level"] = np.where(
            frame["interval_width_95"] <= q50,
            "LOW",
            np.where(frame["interval_width_95"] <= q85, "MEDIUM", "HIGH"),
        )
        return frame

    def calculate_uncertainty_metrics(self, dataframe: pd.DataFrame) -> dict[str, Any]:
        if dataframe.empty:
            return {
                "interval_coverage_95": None,
                "mean_interval_width": None,
                "median_interval_width": None,
                "low_risk_count": 0,
                "medium_risk_count": 0,
                "high_risk_count": 0,
            }
        covered = (
            (dataframe["actual"] >= dataframe["lower_bound_95"])
            & (dataframe["actual"] <= dataframe["upper_bound_95"])
        )
        risk_counts = dataframe["risk_level"].value_counts()
        return {
            "interval_coverage_95": float(covered.mean() * 100),
            "mean_interval_width": float(dataframe["interval_width_95"].mean()),
            "median_interval_width": float(dataframe["interval_width_95"].median()),
            "low_risk_count": int(risk_counts.get("LOW", 0)),
            "medium_risk_count": int(risk_counts.get("MEDIUM", 0)),
            "high_risk_count": int(risk_counts.get("HIGH", 0)),
        }

    def build_xgboost_comparison(
        self,
        prediction_frame: pd.DataFrame,
        gpr_mae: float | int | None,
    ) -> dict[str, Any]:
        xgboost_metrics = calculate_regression_metrics(
            prediction_frame["actual"],
            prediction_frame["xgboost_prediction"],
        )
        xgboost_mae = xgboost_metrics["mae"]
        improvement = (
            (float(xgboost_mae) - float(gpr_mae)) / float(xgboost_mae) * 100
            if xgboost_mae not in (None, 0) and gpr_mae is not None
            else None
        )
        return {
            "xgboost_mae": xgboost_mae,
            "gpr_corrected_mae": gpr_mae,
            "mae_improvement_pct": improvement,
            "comparison_window": "residual_test_period",
        }

    def compare_with_latest_baseline(
        self,
        gpr_mae: float | int | None,
    ) -> tuple[dict[str, Any], str | None]:
        if gpr_mae is None:
            return {}, "GPR corrected MAE is unavailable; baseline comparison skipped"
        with self.session_factory() as session:
            latest_run_id = session.scalar(
                text(
                    """
                    SELECT evaluation_run_id
                    FROM baseline_metrics
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                )
            )
            if latest_run_id is None:
                return {}, "No baseline metrics found for comparison"
            rows = [
                dict(row)
                for row in session.execute(
                    text(
                        """
                        SELECT model_name, mae, created_at
                        FROM baseline_metrics
                        WHERE evaluation_run_id = :evaluation_run_id
                          AND mae IS NOT NULL
                        """
                    ),
                    {"evaluation_run_id": latest_run_id},
                ).mappings()
            ]
        comparison = build_baseline_comparison(rows, float(gpr_mae))
        if not comparison:
            return {}, "No baseline MAE values found for comparison"
        comparison["gpr_corrected_mae"] = comparison.pop("xgboost_mae")
        comparison["baseline_evaluation_run_id"] = latest_run_id
        return comparison, "Baseline comparison may use a different evaluation window."

    def store_predictions(
        self,
        prediction_frame: pd.DataFrame,
        gpr_run_id: str,
        xgboost_training_run_id: str,
        model_version: str,
        session: Session | None = None,
        batch_size: int = 2000,
    ) -> int:
        statement = text(
            """
            INSERT INTO gpr_residual_predictions (
                "timestamp",
                gpr_run_id,
                xgboost_training_run_id,
                model_version,
                xgboost_prediction,
                residual_mean,
                residual_std,
                final_prediction,
                actual,
                lower_bound_95,
                upper_bound_95,
                interval_width_95,
                risk_level,
                error,
                absolute_error,
                percentage_error
            )
            VALUES (
                :timestamp,
                :gpr_run_id,
                :xgboost_training_run_id,
                :model_version,
                :xgboost_prediction,
                :residual_mean,
                :residual_std,
                :final_prediction,
                :actual,
                :lower_bound_95,
                :upper_bound_95,
                :interval_width_95,
                :risk_level,
                :error,
                :absolute_error,
                :percentage_error
            )
            ON CONFLICT ("timestamp", gpr_run_id)
            DO UPDATE SET
                xgboost_prediction = EXCLUDED.xgboost_prediction,
                residual_mean = EXCLUDED.residual_mean,
                residual_std = EXCLUDED.residual_std,
                final_prediction = EXCLUDED.final_prediction,
                actual = EXCLUDED.actual,
                lower_bound_95 = EXCLUDED.lower_bound_95,
                upper_bound_95 = EXCLUDED.upper_bound_95,
                interval_width_95 = EXCLUDED.interval_width_95,
                risk_level = EXCLUDED.risk_level,
                error = EXCLUDED.error,
                absolute_error = EXCLUDED.absolute_error,
                percentage_error = EXCLUDED.percentage_error
            """
        )
        rows = [
            {
                "timestamp": _python_value(row["timestamp"]),
                "gpr_run_id": gpr_run_id,
                "xgboost_training_run_id": xgboost_training_run_id,
                "model_version": model_version,
                "xgboost_prediction": _python_value(row["xgboost_prediction"]),
                "residual_mean": _python_value(row["residual_mean"]),
                "residual_std": _python_value(row["residual_std"]),
                "final_prediction": _python_value(row["final_prediction"]),
                "actual": _python_value(row["actual"]),
                "lower_bound_95": _python_value(row["lower_bound_95"]),
                "upper_bound_95": _python_value(row["upper_bound_95"]),
                "interval_width_95": _python_value(row["interval_width_95"]),
                "risk_level": row["risk_level"],
                "error": _python_value(row["error"]),
                "absolute_error": _python_value(row["absolute_error"]),
                "percentage_error": _python_value(row["percentage_error"]),
            }
            for row in prediction_frame.to_dict(orient="records")
        ]
        return self._execute_batches(statement, rows, session, batch_size)

    def store_metrics(
        self,
        gpr_run_id: str,
        xgboost_training_run_id: str,
        model_version: str,
        residual_train_start: datetime | None,
        residual_train_end: datetime | None,
        residual_test_start: datetime | None,
        residual_test_end: datetime | None,
        train_rows: int,
        test_rows: int,
        max_train_rows: int,
        metrics: dict[str, Any],
        uncertainty_metrics: dict[str, Any],
        xgboost_comparison: dict[str, Any],
        baseline_comparison: dict[str, Any],
        feature_columns: list[str],
        model_params: dict[str, Any],
        artifact_path: str | None,
        warnings: list[str],
        session: Session | None = None,
    ) -> int:
        statement = text(
            """
            INSERT INTO gpr_residual_metrics (
                gpr_run_id,
                xgboost_training_run_id,
                model_version,
                residual_train_start,
                residual_train_end,
                residual_test_start,
                residual_test_end,
                train_rows,
                test_rows,
                max_train_rows,
                mae,
                rmse,
                mape,
                smape,
                r2,
                count,
                mean_actual,
                mean_prediction,
                max_error,
                median_absolute_error,
                interval_coverage_95,
                mean_interval_width,
                median_interval_width,
                low_risk_count,
                medium_risk_count,
                high_risk_count,
                xgboost_comparison,
                baseline_comparison,
                feature_columns,
                model_params,
                artifact_path,
                warnings
            )
            VALUES (
                :gpr_run_id,
                :xgboost_training_run_id,
                :model_version,
                :residual_train_start,
                :residual_train_end,
                :residual_test_start,
                :residual_test_end,
                :train_rows,
                :test_rows,
                :max_train_rows,
                :mae,
                :rmse,
                :mape,
                :smape,
                :r2,
                :count,
                :mean_actual,
                :mean_prediction,
                :max_error,
                :median_absolute_error,
                :interval_coverage_95,
                :mean_interval_width,
                :median_interval_width,
                :low_risk_count,
                :medium_risk_count,
                :high_risk_count,
                CAST(:xgboost_comparison AS JSONB),
                CAST(:baseline_comparison AS JSONB),
                CAST(:feature_columns AS JSONB),
                CAST(:model_params AS JSONB),
                :artifact_path,
                CAST(:warnings AS JSONB)
            )
            ON CONFLICT (gpr_run_id) DO UPDATE SET
                artifact_path = EXCLUDED.artifact_path,
                warnings = EXCLUDED.warnings
            """
        )
        row = {
            "gpr_run_id": gpr_run_id,
            "xgboost_training_run_id": xgboost_training_run_id,
            "model_version": model_version,
            "residual_train_start": residual_train_start,
            "residual_train_end": residual_train_end,
            "residual_test_start": residual_test_start,
            "residual_test_end": residual_test_end,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "max_train_rows": max_train_rows,
            **metrics,
            **uncertainty_metrics,
            "xgboost_comparison": json.dumps(_json_ready(xgboost_comparison)),
            "baseline_comparison": json.dumps(_json_ready(baseline_comparison)),
            "feature_columns": json.dumps(feature_columns),
            "model_params": json.dumps(_json_ready(model_params)),
            "artifact_path": artifact_path,
            "warnings": json.dumps(warnings),
        }
        return self._execute_batches(statement, [row], session, batch_size=1)

    def save_model_artifact(
        self,
        model: Any,
        model_version: str,
        gpr_run_id: str,
    ) -> str:
        artifact_dir = self.artifacts_root / model_version / gpr_run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "model.joblib"
        joblib.dump(model, artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"GPR model artifact was not created at {artifact_path}"
            )
        return artifact_path.as_posix()

    def run_residual_modeling(
        self,
        xgboost_training_run_id: str | None = None,
        residual_train_start: date | datetime | None = None,
        residual_train_end: date | datetime | None = None,
        residual_test_start: date | datetime | None = None,
        residual_test_end: date | datetime | None = None,
        model_version: str = DEFAULT_MODEL_VERSION,
        max_train_rows: int = DEFAULT_MAX_TRAIN_ROWS,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not model_version.strip():
            raise ValueError("model_version must not be empty")
        if max_train_rows <= 0:
            raise ValueError("max_train_rows must be positive")

        warnings: list[str] = [
            "Lagged XGBoost error features are skipped in gpr_residual_v1."
        ]
        errors: list[str] = []
        gpr_run_id = str(uuid.uuid4())
        resolved_xgboost_run_id = (
            xgboost_training_run_id or self.get_latest_successful_xgboost_run()
        )
        if resolved_xgboost_run_id is None:
            return self._summary(
                gpr_run_id,
                "",
                model_version,
                residual_train_start or DEFAULT_RESIDUAL_TRAIN_START,
                residual_train_end or DEFAULT_RESIDUAL_TRAIN_END,
                residual_test_start or DEFAULT_RESIDUAL_TEST_START,
                residual_test_end or DEFAULT_RESIDUAL_TEST_END,
                0,
                0,
                max_train_rows,
                {},
                {},
                {},
                {},
                None,
                warnings,
                ["No successful XGBoost run with an artifact_path was found"],
            )

        train_start = residual_train_start or DEFAULT_RESIDUAL_TRAIN_START
        train_end = residual_train_end or DEFAULT_RESIDUAL_TRAIN_END
        test_start = residual_test_start or DEFAULT_RESIDUAL_TEST_START
        test_end = residual_test_end or DEFAULT_RESIDUAL_TEST_END
        model_params = {**DEFAULT_GPR_PARAMS, **(params or {})}

        residual_data = self.load_residual_data(
            resolved_xgboost_run_id,
            train_start,
            train_end,
            test_start,
            test_end,
        )
        train_frame = residual_data.loc[residual_data["split"] == "train"].copy()
        test_frame = residual_data.loc[residual_data["split"] == "test"].copy()
        train_frame, was_downsampled = self.select_recent_training_window(
            train_frame,
            max_train_rows=max_train_rows,
        )
        if was_downsampled:
            warnings.append(
                f"Residual training rows exceeded max_train_rows; used most recent {max_train_rows} rows."
            )

        (
            X_train,
            y_train,
            train_timestamps,
            _train_xgb_predictions,
            _train_actuals,
            fill_values,
        ) = self.prepare_residual_features(train_frame)
        (
            X_test,
            y_test,
            test_timestamps,
            test_xgb_predictions,
            test_actuals,
            _,
        ) = self.prepare_residual_features(
            test_frame,
            feature_columns=list(X_train.columns),
            fill_values=fill_values,
        )

        if X_train.empty:
            errors.append("No residual training rows available after preparation")
        if X_test.empty:
            errors.append("No residual test rows available after preparation")
        if errors:
            return self._summary(
                gpr_run_id,
                resolved_xgboost_run_id,
                model_version,
                train_start,
                train_end,
                test_start,
                test_end,
                len(X_train),
                len(X_test),
                max_train_rows,
                {},
                {},
                {},
                {},
                None,
                warnings,
                errors,
            )

        model = self.train_gpr_model(X_train, y_train, model_params)
        metrics, prediction_frame = self.evaluate_gpr_residual_model(
            model,
            X_test,
            y_test,
            test_xgb_predictions,
            test_actuals,
            test_timestamps,
        )
        uncertainty_metrics = self.calculate_uncertainty_metrics(prediction_frame)
        xgboost_comparison = self.build_xgboost_comparison(
            prediction_frame,
            metrics.get("mae"),
        )
        baseline_comparison, baseline_warning = self.compare_with_latest_baseline(
            metrics.get("mae")
        )
        if baseline_warning:
            warnings.append(baseline_warning)

        artifact_path = self.save_model_artifact(model, model_version, gpr_run_id)
        with self.session_factory() as session:
            try:
                self.store_predictions(
                    prediction_frame,
                    gpr_run_id,
                    resolved_xgboost_run_id,
                    model_version,
                    session=session,
                )
                self.store_metrics(
                    gpr_run_id=gpr_run_id,
                    xgboost_training_run_id=resolved_xgboost_run_id,
                    model_version=model_version,
                    residual_train_start=_series_min_datetime(train_timestamps),
                    residual_train_end=_series_max_datetime(train_timestamps),
                    residual_test_start=_series_min_datetime(test_timestamps),
                    residual_test_end=_series_max_datetime(test_timestamps),
                    train_rows=len(X_train),
                    test_rows=len(X_test),
                    max_train_rows=max_train_rows,
                    metrics=metrics,
                    uncertainty_metrics=uncertainty_metrics,
                    xgboost_comparison=xgboost_comparison,
                    baseline_comparison=baseline_comparison,
                    feature_columns=list(X_train.columns),
                    model_params=model_params,
                    artifact_path=artifact_path,
                    warnings=warnings,
                    session=session,
                )
                session.commit()
            except SQLAlchemyError as exc:
                session.rollback()
                logger.exception("Could not persist GPR residual results.")
                errors.append(f"Database persistence failed: {exc}")

        mlflow_warning = self._log_to_mlflow(
            gpr_run_id=gpr_run_id,
            xgboost_training_run_id=resolved_xgboost_run_id,
            model_version=model_version,
            train_start=_date_label(train_start),
            train_end=_date_label(train_end),
            test_start=_date_label(test_start),
            test_end=_date_label(test_end),
            train_rows=len(X_train),
            test_rows=len(X_test),
            max_train_rows=max_train_rows,
            metrics=metrics,
            uncertainty_metrics=uncertainty_metrics,
            xgboost_comparison=xgboost_comparison,
            baseline_comparison=baseline_comparison,
            feature_columns=list(X_train.columns),
            model_params=model_params,
            artifact_path=artifact_path,
        )
        if mlflow_warning:
            warnings.append(mlflow_warning)

        return self._summary(
            gpr_run_id,
            resolved_xgboost_run_id,
            model_version,
            train_start,
            train_end,
            test_start,
            test_end,
            len(X_train),
            len(X_test),
            max_train_rows,
            metrics,
            uncertainty_metrics,
            xgboost_comparison,
            baseline_comparison,
            artifact_path,
            warnings,
            errors,
        )

    def get_status(self) -> dict[str, Any]:
        with self.session_factory() as session:
            counts = session.execute(
                text(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM gpr_residual_predictions)
                            AS total_prediction_rows,
                        (SELECT COUNT(*) FROM gpr_residual_metrics)
                            AS total_metric_rows
                    """
                )
            ).mappings().one()
            latest = session.execute(
                text(
                    """
                    SELECT *
                    FROM gpr_residual_metrics
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                )
            ).mappings().one_or_none()
            available_versions = list(
                session.scalars(
                    text(
                        """
                        SELECT DISTINCT model_version
                        FROM gpr_residual_metrics
                        ORDER BY model_version
                        """
                    )
                )
            )
        latest_metrics = None
        latest_uncertainty_metrics = None
        if latest is not None:
            latest_metrics = {
                metric_name: _json_number(latest[metric_name])
                for metric_name in METRIC_NAMES
            }
            latest_uncertainty_metrics = {
                metric_name: _json_number(latest[metric_name])
                for metric_name in UNCERTAINTY_METRIC_NAMES
            }
        return {
            **dict(counts),
            "latest_gpr_run_id": latest["gpr_run_id"] if latest else None,
            "latest_created_at": latest["created_at"] if latest else None,
            "available_model_versions": available_versions,
            "latest_metrics": latest_metrics,
            "latest_uncertainty_metrics": latest_uncertainty_metrics,
            "latest_xgboost_comparison": _json_ready(latest["xgboost_comparison"] or {})
            if latest
            else None,
            "latest_baseline_comparison": _json_ready(latest["baseline_comparison"] or {})
            if latest
            else None,
            "latest_artifact_path": latest["artifact_path"] if latest else None,
        }

    def _execute_batches(
        self,
        statement: Any,
        rows: list[dict[str, Any]],
        session: Session | None,
        batch_size: int,
    ) -> int:
        if not rows:
            return 0
        owns_session = session is None
        database_session = session or self.session_factory()
        affected_rows = 0
        try:
            for offset in range(0, len(rows), batch_size):
                batch = rows[offset : offset + batch_size]
                result = database_session.execute(statement, batch)
                affected_rows += (
                    result.rowcount if result.rowcount >= 0 else len(batch)
                )
            if owns_session:
                database_session.commit()
            return affected_rows
        except SQLAlchemyError:
            if owns_session:
                database_session.rollback()
            raise
        finally:
            if owns_session:
                database_session.close()

    def _log_to_mlflow(
        self,
        gpr_run_id: str,
        xgboost_training_run_id: str,
        model_version: str,
        train_start: str | None,
        train_end: str | None,
        test_start: str | None,
        test_end: str | None,
        train_rows: int,
        test_rows: int,
        max_train_rows: int,
        metrics: dict[str, Any],
        uncertainty_metrics: dict[str, Any],
        xgboost_comparison: dict[str, Any],
        baseline_comparison: dict[str, Any],
        feature_columns: list[str],
        model_params: dict[str, Any],
        artifact_path: str | None,
    ) -> str | None:
        try:
            import mlflow

            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(
                run_name=f"gpr-residual-{gpr_run_id[:8]}",
                tags={
                    "gpr_run_id": gpr_run_id,
                    "xgboost_training_run_id": xgboost_training_run_id,
                    "model_version": model_version,
                },
            ):
                mlflow.log_params(
                    {
                        **model_params,
                        "model_version": model_version,
                        "train_start": train_start or "",
                        "train_end": train_end or "",
                        "test_start": test_start or "",
                        "test_end": test_end or "",
                        "train_rows": train_rows,
                        "test_rows": test_rows,
                        "max_train_rows": max_train_rows,
                        "feature_count": len(feature_columns),
                    }
                )
                for metric_name, metric_value in {
                    **metrics,
                    **uncertainty_metrics,
                }.items():
                    if metric_value is not None:
                        mlflow.log_metric(metric_name, float(metric_value))
                if xgboost_comparison.get("mae_improvement_pct") is not None:
                    mlflow.log_metric(
                        "xgboost_mae_improvement_pct",
                        float(xgboost_comparison["mae_improvement_pct"]),
                    )
                mlflow.log_text(
                    json.dumps(feature_columns, indent=2),
                    "feature_columns.json",
                )
                mlflow.log_text(
                    json.dumps(_json_ready(xgboost_comparison), indent=2),
                    "xgboost_comparison.json",
                )
                mlflow.log_text(
                    json.dumps(_json_ready(baseline_comparison), indent=2),
                    "baseline_comparison.json",
                )
                if artifact_path is not None:
                    mlflow.log_artifact(artifact_path, artifact_path="model")
            return None
        except Exception as exc:
            logger.warning("MLflow GPR residual logging failed: %s", exc)
            return f"MLflow logging failed: {exc}"

    def _summary(
        self,
        gpr_run_id: str,
        xgboost_training_run_id: str,
        model_version: str,
        residual_train_start: date | datetime,
        residual_train_end: date | datetime,
        residual_test_start: date | datetime,
        residual_test_end: date | datetime,
        train_rows: int,
        test_rows: int,
        max_train_rows: int,
        metrics: dict[str, Any],
        uncertainty_metrics: dict[str, Any],
        xgboost_comparison: dict[str, Any],
        baseline_comparison: dict[str, Any],
        artifact_path: str | None,
        warnings: list[str],
        errors: list[str],
    ) -> dict[str, Any]:
        return {
            "gpr_run_id": gpr_run_id,
            "xgboost_training_run_id": xgboost_training_run_id,
            "model_version": model_version,
            "residual_train_start": _date_label(residual_train_start),
            "residual_train_end": _date_label(residual_train_end),
            "residual_test_start": _date_label(residual_test_start),
            "residual_test_end": _date_label(residual_test_end),
            "train_rows": train_rows,
            "test_rows": test_rows,
            "max_train_rows": max_train_rows,
            "metrics": _json_ready(metrics),
            "uncertainty_metrics": _json_ready(uncertainty_metrics),
            "xgboost_comparison": _json_ready(xgboost_comparison),
            "baseline_comparison": _json_ready(baseline_comparison),
            "artifact_path": artifact_path,
            "warnings": warnings,
            "errors": errors,
        }


def _normalize_boundary(value: date | datetime, is_end: bool) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=ISTANBUL_TIMEZONE)
        return value.astimezone(ISTANBUL_TIMEZONE)
    boundary = time(23, 59, 59) if is_end else time.min
    return datetime.combine(value, boundary, tzinfo=ISTANBUL_TIMEZONE)


def _date_label(value: date | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        normalized = (
            value.replace(tzinfo=ISTANBUL_TIMEZONE)
            if value.tzinfo is None
            else value.astimezone(ISTANBUL_TIMEZONE)
        )
        return normalized.date().isoformat()
    return value.isoformat()


def _finite_median(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return 0.0
    return float(numeric.median())


def _series_min_datetime(series: pd.Series) -> datetime | None:
    if series.empty:
        return None
    return _python_value(series.min())


def _series_max_datetime(series: pd.Series) -> datetime | None:
    if series.empty:
        return None
    return _python_value(series.max())


def _python_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_number(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value
