import json
import logging
import uuid
from collections.abc import Callable
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal, engine
from ml.evaluation.metrics import calculate_regression_metrics

logger = logging.getLogger(__name__)

ISTANBUL_TIMEZONE = ZoneInfo("Europe/Istanbul")
DEFAULT_TRAIN_START = date(2020, 1, 1)
DEFAULT_TRAIN_END = date(2023, 12, 31)
DEFAULT_TEST_START = date(2024, 1, 1)
DEFAULT_MODEL_VERSION = "xgboost_v1"
MLFLOW_EXPERIMENT_NAME = "ptf_xgboost_forecasting"
DEFAULT_XGBOOST_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}
EXCLUDED_FEATURE_COLUMNS = {
    "timestamp",
    "target_ptf",
    "feature_version",
    "created_at",
    "updated_at",
    # These engineered columns are calculated from the current target_ptf value.
    # They are safe to store for diagnostics, but not safe as same-hour model inputs.
    "ptf_diff_1",
    "ptf_diff_24",
    "ptf_pct_change_1",
    "ptf_pct_change_24",
}
REQUIRED_NON_NULL_FEATURES = (
    "ptf_lag_24",
    "ptf_lag_168",
    "ptf_24h_mean",
    "ptf_7d_mean",
)
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


class XGBoostPtfService:
    def __init__(
        self,
        session_factory: Callable[[], Session] = SessionLocal,
        mlflow_tracking_uri: str | None = None,
        artifacts_root: Path | str = "artifacts/models/ptf/xgboost",
    ) -> None:
        self.session_factory = session_factory
        self.mlflow_tracking_uri = mlflow_tracking_uri or settings.mlflow_tracking_uri
        self.artifacts_root = Path(artifacts_root)

    def load_training_data(
        self,
        train_start: date | datetime,
        train_end: date | datetime,
        test_start: date | datetime,
        test_end: date | datetime | None,
        feature_version: str = "v1",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_start_ts = _normalize_boundary(train_start, is_end=False)
        train_end_ts = _normalize_boundary(train_end, is_end=True)
        test_start_ts = _normalize_boundary(test_start, is_end=False)
        test_end_ts = (
            _normalize_boundary(test_end, is_end=True)
            if test_end is not None
            else None
        )
        if train_end_ts < train_start_ts:
            raise ValueError("train_end must be on or after train_start")
        if test_end_ts is not None and test_end_ts < test_start_ts:
            raise ValueError("test_end must be on or after test_start")

        train_frame = self._load_feature_slice(
            start_timestamp=train_start_ts,
            end_timestamp=train_end_ts,
            feature_version=feature_version,
        )
        test_frame = self._load_feature_slice(
            start_timestamp=test_start_ts,
            end_timestamp=test_end_ts,
            feature_version=feature_version,
        )
        return train_frame, test_frame

    def prepare_features(
        self,
        dataframe: pd.DataFrame,
        feature_columns: list[str] | None = None,
        fill_values: dict[str, float] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, float]]:
        required = {"timestamp", "target_ptf"}
        missing = required.difference(dataframe.columns)
        if missing:
            raise ValueError(
                f"Missing required training columns: {', '.join(sorted(missing))}"
            )

        frame = dataframe.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["target_ptf"] = pd.to_numeric(frame["target_ptf"], errors="coerce")
        frame = frame.sort_values("timestamp").reset_index(drop=True)

        non_null_subset = ["target_ptf"]
        available_required_features = [
            column for column in REQUIRED_NON_NULL_FEATURES if column in frame.columns
        ]
        non_null_subset.extend(available_required_features)
        frame = frame.dropna(subset=non_null_subset).reset_index(drop=True)

        target = frame["target_ptf"].astype(float)
        timestamps = frame["timestamp"]
        feature_frame = self._build_model_feature_frame(frame)

        if feature_columns is None:
            feature_columns = list(feature_frame.columns)
        feature_frame = feature_frame.reindex(columns=feature_columns)

        if fill_values is None:
            fill_values = {
                column: _finite_median(feature_frame[column])
                for column in feature_columns
            }
        feature_frame = feature_frame.fillna(fill_values).fillna(0.0)
        feature_frame = feature_frame.astype(float)

        finite_rows = (
            np.isfinite(feature_frame.to_numpy(dtype=float)).all(axis=1)
            & np.isfinite(target.to_numpy(dtype=float))
        )
        feature_frame = feature_frame.loc[finite_rows].reset_index(drop=True)
        target = target.loc[finite_rows].reset_index(drop=True)
        timestamps = timestamps.loc[finite_rows].reset_index(drop=True)
        return feature_frame, target, timestamps, fill_values

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: dict[str, Any] | None = None,
    ) -> Any:
        if X_train.empty or y_train.empty:
            raise ValueError("Training data is empty after feature preparation")
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError(
                "xgboost is not installed. Rebuild the API image after updating "
                "requirements.txt."
            ) from exc

        model_params = {**DEFAULT_XGBOOST_PARAMS, **(params or {})}
        model = XGBRegressor(**model_params)
        model.fit(X_train, y_train)
        return model

    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        timestamps: pd.Series,
    ) -> tuple[dict[str, float | int | None], pd.DataFrame]:
        if X_test.empty or y_test.empty:
            raise ValueError("Test data is empty after feature preparation")
        predictions = model.predict(X_test)
        prediction_frame = pd.DataFrame(
            {
                "timestamp": timestamps.reset_index(drop=True),
                "actual": y_test.reset_index(drop=True),
                "prediction": predictions,
            }
        )
        prediction_frame["error"] = (
            prediction_frame["actual"] - prediction_frame["prediction"]
        )
        prediction_frame["absolute_error"] = prediction_frame["error"].abs()
        denominator = prediction_frame["actual"].abs().replace(0, np.nan)
        prediction_frame["percentage_error"] = (
            prediction_frame["absolute_error"] / denominator * 100
        )
        metrics = calculate_regression_metrics(
            prediction_frame["actual"],
            prediction_frame["prediction"],
        )
        return metrics, prediction_frame

    def store_predictions(
        self,
        prediction_frame: pd.DataFrame,
        training_run_id: str,
        model_version: str,
        session: Session | None = None,
        batch_size: int = 2000,
    ) -> int:
        statement = text(
            """
            INSERT INTO xgboost_predictions (
                "timestamp",
                model_version,
                prediction,
                actual,
                error,
                absolute_error,
                percentage_error,
                training_run_id
            )
            VALUES (
                :timestamp,
                :model_version,
                :prediction,
                :actual,
                :error,
                :absolute_error,
                :percentage_error,
                :training_run_id
            )
            ON CONFLICT ("timestamp", model_version, training_run_id)
            DO UPDATE SET
                prediction = EXCLUDED.prediction,
                actual = EXCLUDED.actual,
                error = EXCLUDED.error,
                absolute_error = EXCLUDED.absolute_error,
                percentage_error = EXCLUDED.percentage_error
            """
        )
        rows = [
            {
                "timestamp": _python_value(row["timestamp"]),
                "model_version": model_version,
                "prediction": _python_value(row["prediction"]),
                "actual": _python_value(row["actual"]),
                "error": _python_value(row["error"]),
                "absolute_error": _python_value(row["absolute_error"]),
                "percentage_error": _python_value(row["percentage_error"]),
                "training_run_id": training_run_id,
            }
            for row in prediction_frame.to_dict(orient="records")
        ]
        return self._execute_batches(statement, rows, session, batch_size)

    def store_metrics(
        self,
        metrics: dict[str, float | int | None],
        training_run_id: str,
        model_version: str,
        train_start: datetime | None,
        train_end: datetime | None,
        test_start: datetime | None,
        test_end: datetime | None,
        baseline_comparison: dict[str, Any],
        feature_columns: list[str],
        model_params: dict[str, Any],
        artifact_path: str | None,
        session: Session | None = None,
    ) -> int:
        statement = text(
            """
            INSERT INTO xgboost_metrics (
                training_run_id,
                model_version,
                train_start,
                train_end,
                test_start,
                test_end,
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
                baseline_comparison,
                feature_columns,
                model_params,
                artifact_path
            )
            VALUES (
                :training_run_id,
                :model_version,
                :train_start,
                :train_end,
                :test_start,
                :test_end,
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
                CAST(:baseline_comparison AS JSONB),
                CAST(:feature_columns AS JSONB),
                CAST(:model_params AS JSONB),
                :artifact_path
            )
            ON CONFLICT (training_run_id, model_version)
            DO UPDATE SET
                train_start = EXCLUDED.train_start,
                train_end = EXCLUDED.train_end,
                test_start = EXCLUDED.test_start,
                test_end = EXCLUDED.test_end,
                mae = EXCLUDED.mae,
                rmse = EXCLUDED.rmse,
                mape = EXCLUDED.mape,
                smape = EXCLUDED.smape,
                r2 = EXCLUDED.r2,
                count = EXCLUDED.count,
                mean_actual = EXCLUDED.mean_actual,
                mean_prediction = EXCLUDED.mean_prediction,
                max_error = EXCLUDED.max_error,
                median_absolute_error = EXCLUDED.median_absolute_error,
                baseline_comparison = EXCLUDED.baseline_comparison,
                feature_columns = EXCLUDED.feature_columns,
                model_params = EXCLUDED.model_params,
                artifact_path = EXCLUDED.artifact_path
            """
        )
        row = {
            "training_run_id": training_run_id,
            "model_version": model_version,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            **metrics,
            "baseline_comparison": json.dumps(_json_ready(baseline_comparison)),
            "feature_columns": json.dumps(feature_columns),
            "model_params": json.dumps(_json_ready(model_params)),
            "artifact_path": artifact_path,
        }
        return self._execute_batches(statement, [row], session, batch_size=1)

    def save_model_artifact(
        self,
        model: Any,
        model_version: str,
        training_run_id: str,
    ) -> str:
        artifact_dir = self.artifacts_root / model_version / training_run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "model.json"
        native_booster = getattr(model, "_Booster", None)
        if native_booster is not None and hasattr(native_booster, "save_model"):
            native_booster.save_model(str(artifact_path))
        elif hasattr(model, "save_model"):
            model.save_model(str(artifact_path))
        else:
            raise TypeError("Model object does not expose a native save_model method")

        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact was not created at {artifact_path}"
            )
        return artifact_path.as_posix()

    def run_training(
        self,
        train_start: date | datetime | None = None,
        train_end: date | datetime | None = None,
        test_start: date | datetime | None = None,
        test_end: date | datetime | None = None,
        model_version: str = DEFAULT_MODEL_VERSION,
        feature_version: str = "v1",
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not model_version.strip():
            raise ValueError("model_version must not be empty")
        if not feature_version.strip():
            raise ValueError("feature_version must not be empty")

        resolved_train_start = train_start or DEFAULT_TRAIN_START
        resolved_train_end = train_end or DEFAULT_TRAIN_END
        resolved_test_start = test_start or DEFAULT_TEST_START
        model_params = {**DEFAULT_XGBOOST_PARAMS, **(params or {})}
        training_run_id = str(uuid.uuid4())
        warnings: list[str] = []
        errors: list[str] = []

        train_frame, test_frame = self.load_training_data(
            train_start=resolved_train_start,
            train_end=resolved_train_end,
            test_start=resolved_test_start,
            test_end=test_end,
            feature_version=feature_version,
        )
        X_train, y_train, train_timestamps, fill_values = self.prepare_features(
            train_frame
        )
        X_test, y_test, test_timestamps, _ = self.prepare_features(
            test_frame,
            feature_columns=list(X_train.columns),
            fill_values=fill_values,
        )

        if X_train.empty:
            errors.append("No training rows available after feature preparation")
        if X_test.empty:
            errors.append("No test rows available after feature preparation")
        if errors:
            return self._summary(
                training_run_id=training_run_id,
                model_version=model_version,
                train_start=resolved_train_start,
                train_end=resolved_train_end,
                test_start=resolved_test_start,
                test_end=test_end,
                train_rows=len(X_train),
                test_rows=len(X_test),
                metrics={},
                baseline_comparison={},
                artifact_path=None,
                warnings=warnings,
                errors=errors,
            )

        model = self.train_model(X_train, y_train, model_params)
        metrics, prediction_frame = self.evaluate_model(
            model,
            X_test,
            y_test,
            test_timestamps,
        )
        baseline_comparison, comparison_warning = self.compare_with_latest_baseline(
            metrics.get("mae")
        )
        if comparison_warning:
            warnings.append(comparison_warning)

        artifact_path: str | None = None
        try:
            artifact_path = self.save_model_artifact(
                model,
                model_version,
                training_run_id,
            )
        except Exception as exc:
            logger.exception("Could not save XGBoost model artifact.")
            warnings.append(f"Model artifact save failed: {exc}")

        with self.session_factory() as session:
            try:
                self.store_predictions(
                    prediction_frame,
                    training_run_id=training_run_id,
                    model_version=model_version,
                    session=session,
                )
                self.store_metrics(
                    metrics=metrics,
                    training_run_id=training_run_id,
                    model_version=model_version,
                    train_start=_series_min_datetime(train_timestamps),
                    train_end=_series_max_datetime(train_timestamps),
                    test_start=_series_min_datetime(test_timestamps),
                    test_end=_series_max_datetime(test_timestamps),
                    baseline_comparison=baseline_comparison,
                    feature_columns=list(X_train.columns),
                    model_params=model_params,
                    artifact_path=artifact_path,
                    session=session,
                )
                session.commit()
            except SQLAlchemyError as exc:
                session.rollback()
                logger.exception("Could not persist XGBoost training results.")
                errors.append(f"Database persistence failed: {exc}")

        mlflow_warning = self._log_to_mlflow(
            model=model,
            training_run_id=training_run_id,
            model_version=model_version,
            feature_version=feature_version,
            train_start=_date_label(resolved_train_start),
            train_end=_date_label(resolved_train_end),
            test_start=_date_label(resolved_test_start),
            test_end=_date_label(test_end)
            if test_end is not None
            else _date_label(_series_max_datetime(test_timestamps)),
            model_params=model_params,
            metrics=metrics,
            baseline_comparison=baseline_comparison,
            feature_columns=list(X_train.columns),
            artifact_path=artifact_path,
        )
        if mlflow_warning:
            warnings.append(mlflow_warning)

        return self._summary(
            training_run_id=training_run_id,
            model_version=model_version,
            train_start=resolved_train_start,
            train_end=resolved_train_end,
            test_start=resolved_test_start,
            test_end=test_end or _series_max_datetime(test_timestamps),
            train_rows=len(X_train),
            test_rows=len(X_test),
            metrics=metrics,
            baseline_comparison=baseline_comparison,
            artifact_path=artifact_path,
            warnings=warnings,
            errors=errors,
        )

    def compare_with_latest_baseline(
        self,
        xgboost_mae: float | int | None,
    ) -> tuple[dict[str, Any], str | None]:
        if xgboost_mae is None:
            return {}, "XGBoost MAE is unavailable; baseline comparison skipped"
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
        comparison = build_baseline_comparison(rows, float(xgboost_mae))
        if not comparison:
            return {}, "No baseline MAE values found for comparison"
        comparison["baseline_evaluation_run_id"] = latest_run_id
        return comparison, None

    def get_status(self) -> dict[str, Any]:
        with self.session_factory() as session:
            counts = session.execute(
                text(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM xgboost_predictions)
                            AS total_prediction_rows,
                        (SELECT COUNT(*) FROM xgboost_metrics)
                            AS total_metric_rows
                    """
                )
            ).mappings().one()
            latest = session.execute(
                text(
                    """
                    SELECT *
                    FROM xgboost_metrics
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
                        FROM xgboost_metrics
                        ORDER BY model_version
                        """
                    )
                )
            )

        latest_metrics: dict[str, Any] | None = None
        latest_baseline_comparison: dict[str, Any] | None = None
        if latest is not None:
            latest_metrics = {
                metric_name: _json_number(latest[metric_name])
                for metric_name in METRIC_NAMES
            }
            latest_baseline_comparison = _json_ready(
                latest["baseline_comparison"] or {}
            )

        return {
            **dict(counts),
            "latest_training_run_id": latest["training_run_id"] if latest else None,
            "latest_created_at": latest["created_at"] if latest else None,
            "available_model_versions": available_versions,
            "latest_metrics": latest_metrics,
            "latest_baseline_comparison": latest_baseline_comparison,
        }

    def _load_feature_slice(
        self,
        start_timestamp: datetime,
        end_timestamp: datetime | None,
        feature_version: str,
    ) -> pd.DataFrame:
        clauses = ['"timestamp" >= :start_timestamp', "feature_version = :feature_version"]
        parameters: dict[str, Any] = {
            "start_timestamp": start_timestamp,
            "feature_version": feature_version,
        }
        if end_timestamp is not None:
            clauses.append('"timestamp" <= :end_timestamp')
            parameters["end_timestamp"] = end_timestamp
        query = text(
            f"""
            SELECT *
            FROM features_ptf_hourly
            WHERE {' AND '.join(clauses)}
            ORDER BY "timestamp"
            """
        )
        with engine.connect() as connection:
            dataframe = pd.read_sql_query(query, connection, params=parameters)
        if not dataframe.empty:
            dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=True)
        return dataframe

    def _build_model_feature_frame(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        feature_parts: list[pd.DataFrame] = []
        for column in dataframe.columns:
            if column in EXCLUDED_FEATURE_COLUMNS:
                continue
            if column == "season":
                season_dummies = pd.get_dummies(
                    dataframe[column].fillna("unknown").astype(str),
                    prefix="season",
                    dtype=int,
                )
                feature_parts.append(season_dummies)
                continue
            series = dataframe[column]
            if pd.api.types.is_bool_dtype(series):
                feature_parts.append(series.astype(int).to_frame(column))
                continue
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                feature_parts.append(numeric.to_frame(column))

        if not feature_parts:
            raise ValueError("No usable model feature columns were found")
        feature_frame = pd.concat(feature_parts, axis=1)
        return feature_frame.reindex(sorted(feature_frame.columns), axis=1)

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
        model: Any,
        training_run_id: str,
        model_version: str,
        feature_version: str,
        train_start: str | None,
        train_end: str | None,
        test_start: str | None,
        test_end: str | None,
        model_params: dict[str, Any],
        metrics: dict[str, float | int | None],
        baseline_comparison: dict[str, Any],
        feature_columns: list[str],
        artifact_path: str | None,
    ) -> str | None:
        try:
            import mlflow

            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(
                run_name=f"xgboost-{training_run_id[:8]}",
                tags={
                    "training_run_id": training_run_id,
                    "model_version": model_version,
                },
            ):
                mlflow.log_params(
                    {
                        **model_params,
                        "model_version": model_version,
                        "feature_version": feature_version,
                        "train_start": train_start or "",
                        "train_end": train_end or "",
                        "test_start": test_start or "",
                        "test_end": test_end or "",
                        "feature_count": len(feature_columns),
                    }
                )
                for metric_name, metric_value in metrics.items():
                    if metric_value is not None:
                        mlflow.log_metric(metric_name, float(metric_value))
                if baseline_comparison.get("mae_improvement_pct") is not None:
                    mlflow.log_metric(
                        "baseline_mae_improvement_pct",
                        float(baseline_comparison["mae_improvement_pct"]),
                    )
                mlflow.log_text(
                    json.dumps(feature_columns, indent=2),
                    "feature_columns.json",
                )
                mlflow.log_text(
                    json.dumps(_json_ready(baseline_comparison), indent=2),
                    "baseline_comparison.json",
                )
                if artifact_path is not None:
                    mlflow.log_artifact(artifact_path, artifact_path="model")
            return None
        except Exception as exc:
            logger.warning("MLflow XGBoost logging failed: %s", exc)
            return f"MLflow logging failed: {exc}"

    def _summary(
        self,
        training_run_id: str,
        model_version: str,
        train_start: date | datetime,
        train_end: date | datetime,
        test_start: date | datetime,
        test_end: date | datetime | None,
        train_rows: int,
        test_rows: int,
        metrics: dict[str, Any],
        baseline_comparison: dict[str, Any],
        artifact_path: str | None,
        warnings: list[str],
        errors: list[str],
    ) -> dict[str, Any]:
        return {
            "training_run_id": training_run_id,
            "model_version": model_version,
            "train_start": _date_label(train_start),
            "train_end": _date_label(train_end),
            "test_start": _date_label(test_start),
            "test_end": _date_label(test_end),
            "train_rows": train_rows,
            "test_rows": test_rows,
            "metrics": _json_ready(metrics),
            "baseline_comparison": _json_ready(baseline_comparison),
            "artifact_path": artifact_path,
            "warnings": warnings,
            "errors": errors,
        }


def build_baseline_comparison(
    baseline_rows: list[dict[str, Any]],
    xgboost_mae: float,
) -> dict[str, Any]:
    valid_rows = [
        row
        for row in baseline_rows
        if row.get("mae") is not None and _json_number(row["mae"]) is not None
    ]
    if not valid_rows:
        return {}
    best = min(valid_rows, key=lambda row: float(_json_number(row["mae"])))
    best_mae = float(_json_number(best["mae"]))
    improvement = (
        (best_mae - xgboost_mae) / best_mae * 100 if best_mae != 0 else None
    )
    return {
        "best_baseline_model": best["model_name"],
        "best_baseline_mae": best_mae,
        "xgboost_mae": float(xgboost_mae),
        "mae_improvement_pct": improvement,
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
    value = series.min()
    return _python_value(value)


def _series_max_datetime(series: pd.Series) -> datetime | None:
    if series.empty:
        return None
    value = series.max()
    return _python_value(value)


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
