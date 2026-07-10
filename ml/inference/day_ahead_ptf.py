import json
import logging
import shutil
import uuid
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
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

from app.db.session import SessionLocal, engine
from app.core.config import settings
from ml.models.gpr_residual_ptf import GPR_FEATURE_COLUMNS
from ml.models.xgboost_ptf import EXCLUDED_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

ISTANBUL_TIMEZONE = ZoneInfo("Europe/Istanbul")
DEFAULT_MODEL_VERSION = "day_ahead_v1"
DEFAULT_HORIZON_HOURS = 24
MAX_HISTORY_HOURS = 30 * 24 + 168 + 24
LAG_HOURS = (1, 2, 3, 24, 48, 72, 168)
GENERATION_METHOD = "latest_history_calendar_lag_rolling_v1"


class DayAheadPtfForecastService:
    def __init__(
        self,
        session_factory: Callable[[], Session] = SessionLocal,
        xgboost_artifacts_root: Path | str = "artifacts/models/ptf/xgboost",
        gpr_artifacts_root: Path | str = "artifacts/models/ptf/gpr_residual",
    ) -> None:
        self.session_factory = session_factory
        self.xgboost_artifacts_root = Path(xgboost_artifacts_root)
        self.gpr_artifacts_root = Path(gpr_artifacts_root)

    def get_latest_xgboost_artifact(self) -> dict[str, Any] | None:
        with self.session_factory() as session:
            row = session.execute(
                text(
                    """
                    SELECT
                        training_run_id,
                        model_version,
                        feature_columns,
                        artifact_path,
                        created_at
                    FROM xgboost_metrics
                    WHERE artifact_path IS NOT NULL
                      AND artifact_path <> ''
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                )
            ).mappings().one_or_none()
        return _mapping_to_dict(row)

    def get_latest_gpr_artifact(self) -> dict[str, Any] | None:
        with self.session_factory() as session:
            row = session.execute(
                text(
                    """
                    SELECT
                        gpr_run_id,
                        xgboost_training_run_id,
                        model_version,
                        feature_columns,
                        artifact_path,
                        mean_interval_width,
                        median_interval_width,
                        created_at
                    FROM gpr_residual_metrics
                    WHERE artifact_path IS NOT NULL
                      AND artifact_path <> ''
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                )
            ).mappings().one_or_none()
        return _mapping_to_dict(row)

    def get_latest_decision_metadata(self) -> dict[str, Any] | None:
        with self.session_factory() as session:
            row = session.execute(
                text(
                    """
                    SELECT
                        decision_run_id,
                        gpr_run_id,
                        xgboost_training_run_id,
                        model_version,
                        selected_model,
                        mean_interval_width,
                        median_interval_width,
                        created_at
                    FROM forecast_decision_metrics
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                )
            ).mappings().one_or_none()
        return _mapping_to_dict(row)

    def load_recent_ptf_history(self, max_history_hours: int = MAX_HISTORY_HOURS) -> pd.DataFrame:
        query = text(
            """
            SELECT "timestamp", ptf_tl
            FROM (
                SELECT "timestamp", ptf_tl
                FROM ptf_hourly
                WHERE ptf_tl IS NOT NULL
                ORDER BY "timestamp" DESC
                LIMIT :limit
            ) AS recent
            ORDER BY "timestamp"
            """
        )
        with engine.connect() as connection:
            dataframe = pd.read_sql_query(
                query,
                connection,
                params={"limit": int(max_history_hours)},
            )
        if dataframe.empty:
            return pd.DataFrame(
                {
                    "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                    "ptf_tl": pd.Series(dtype="float64"),
                }
            )
        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=True)
        dataframe["ptf_tl"] = pd.to_numeric(dataframe["ptf_tl"], errors="coerce")
        return dataframe.dropna(subset=["timestamp", "ptf_tl"]).sort_values(
            "timestamp"
        ).reset_index(drop=True)

    def build_future_feature_frame(
        self,
        target_date: date | None = None,
        horizon_hours: int = DEFAULT_HORIZON_HOURS,
        history: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        if horizon_hours <= 0:
            raise ValueError("horizon_hours must be positive")
        warnings: list[str] = []
        ptf_history = history if history is not None else self.load_recent_ptf_history()
        if ptf_history.empty:
            raise ValueError("No PTF history found in ptf_hourly")

        ptf_history = ptf_history.copy()
        ptf_history["timestamp"] = pd.to_datetime(ptf_history["timestamp"], utc=True)
        ptf_history["ptf_tl"] = pd.to_numeric(ptf_history["ptf_tl"], errors="coerce")
        ptf_history = ptf_history.dropna(subset=["timestamp", "ptf_tl"]).sort_values(
            "timestamp"
        )
        if ptf_history.empty:
            raise ValueError("PTF history contains no usable numeric ptf_tl values")

        latest_timestamp = pd.Timestamp(ptf_history["timestamp"].max()).tz_convert(
            ISTANBUL_TIMEZONE
        )
        resolved_target_date = target_date or (latest_timestamp.date() + timedelta(days=1))
        future_timestamps = build_target_timestamps(resolved_target_date, horizon_hours)

        if future_timestamps[0] > latest_timestamp + timedelta(hours=24):
            warnings.append(
                "Latest PTF history is more than 24 hours before the target horizon; "
                "forecast uses stale lag and rolling features."
            )

        history_by_timestamp = {
            pd.Timestamp(row["timestamp"]).tz_convert("UTC"): float(row["ptf_tl"])
            for row in ptf_history.to_dict(orient="records")
        }
        latest_value = float(ptf_history["ptf_tl"].iloc[-1])
        history_values = ptf_history.set_index("timestamp")["ptf_tl"].astype(float)

        records: list[dict[str, Any]] = []
        used_lag_fallbacks: set[int] = set()
        used_rolling_fallbacks: set[str] = set()
        for target_timestamp in future_timestamps:
            target_utc = target_timestamp.tz_convert("UTC")
            record = _calendar_feature_record(target_timestamp)
            record["timestamp"] = target_timestamp
            record["target_ptf"] = np.nan

            for lag in LAG_HOURS:
                lag_timestamp = target_utc - timedelta(hours=lag)
                value = history_by_timestamp.get(lag_timestamp)
                if value is None:
                    value = latest_value
                    used_lag_fallbacks.add(lag)
                record[f"ptf_lag_{lag}"] = value

            previous_values = history_values[history_values.index < target_utc]
            if previous_values.empty:
                previous_values = history_values
            _add_future_rolling_features(
                record,
                previous_values,
                used_rolling_fallbacks,
            )
            record["ptf_diff_1"] = np.nan
            record["ptf_diff_24"] = np.nan
            record["ptf_pct_change_1"] = np.nan
            record["ptf_pct_change_24"] = np.nan
            record["feature_version"] = "future_v1"
            records.append(record)

        if used_lag_fallbacks:
            warnings.append(
                "Used latest known PTF value for unavailable future lag features: "
                + ", ".join(f"ptf_lag_{lag}" for lag in sorted(used_lag_fallbacks))
            )
        if used_rolling_fallbacks:
            warnings.append(
                "Some rolling features used shorter available history windows: "
                + ", ".join(sorted(used_rolling_fallbacks))
            )
        warnings.append(
            "Sprint 9 MVP does not recursively feed predicted PTF values into future lag features."
        )

        return pd.DataFrame(records), warnings

    def load_xgboost_model(self, artifact_path: str) -> Any:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed in the API image") from exc

        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"XGBoost artifact not found at {artifact_path}")
        model = XGBRegressor()
        model.load_model(str(path))
        return model

    def load_gpr_model(self, artifact_path: str) -> Any:
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"GPR artifact not found at {artifact_path}")
        return joblib.load(path)

    def ensure_xgboost_artifact(
        self,
        metadata: dict[str, Any],
        warnings: list[str],
    ) -> str:
        artifact_path = Path(str(metadata["artifact_path"]))
        if artifact_path.exists():
            return artifact_path.as_posix()
        recovered = self._recover_mlflow_artifact(
            tag_key="training_run_id",
            tag_value=str(metadata["training_run_id"]),
            artifact_relative_path="model/model.json",
            target_path=artifact_path,
        )
        if recovered:
            warnings.append(
                "Recovered missing local XGBoost artifact from MLflow artifact store."
            )
        return artifact_path.as_posix()

    def ensure_gpr_artifact(
        self,
        metadata: dict[str, Any],
        warnings: list[str],
    ) -> str:
        artifact_path = Path(str(metadata["artifact_path"]))
        if artifact_path.exists():
            return artifact_path.as_posix()
        recovered = self._recover_mlflow_artifact(
            tag_key="gpr_run_id",
            tag_value=str(metadata["gpr_run_id"]),
            artifact_relative_path="model/model.joblib",
            target_path=artifact_path,
        )
        if recovered:
            warnings.append(
                "Recovered missing local GPR artifact from MLflow artifact store."
            )
        return artifact_path.as_posix()

    def _recover_mlflow_artifact(
        self,
        tag_key: str,
        tag_value: str,
        artifact_relative_path: str,
        target_path: Path,
    ) -> bool:
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
            experiments = client.search_experiments()
            for experiment in experiments:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"tags.{tag_key} = '{tag_value}'",
                    max_results=1,
                    order_by=["attribute.start_time DESC"],
                )
                if not runs:
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)
                downloaded_path = Path(
                    client.download_artifacts(
                        runs[0].info.run_id,
                        artifact_relative_path,
                        dst_path=str(target_path.parent),
                    )
                )
                shutil.copyfile(downloaded_path, target_path)
                return target_path.exists()
        except Exception as exc:
            logger.warning("MLflow artifact recovery failed: %s", exc)
        return False

    def align_feature_frame(
        self,
        dataframe: pd.DataFrame,
        feature_columns: list[str],
    ) -> tuple[pd.DataFrame, list[str]]:
        feature_frame = _build_numeric_feature_frame(dataframe)
        missing_columns = [
            column for column in feature_columns if column not in feature_frame.columns
        ]
        extra_columns = [
            column for column in feature_frame.columns if column not in feature_columns
        ]
        aligned = feature_frame.reindex(columns=feature_columns)
        aligned = aligned.apply(pd.to_numeric, errors="coerce")
        fill_values = {
            column: _finite_median(aligned[column])
            for column in aligned.columns
        }
        aligned = aligned.fillna(fill_values).fillna(0.0).astype(float)
        warnings: list[str] = []
        if missing_columns:
            warnings.append(
                "Filled missing model feature columns with safe defaults: "
                + ", ".join(missing_columns)
            )
        if extra_columns:
            warnings.append(
                "Ignored feature columns not used by the trained model: "
                + ", ".join(extra_columns)
            )
        return aligned, warnings

    def predict_xgboost(self, model: Any, features: pd.DataFrame) -> np.ndarray:
        predictions = model.predict(features)
        return np.asarray(predictions, dtype=float)

    def predict_uncertainty(
        self,
        future_features: pd.DataFrame,
        xgboost_predictions: np.ndarray,
        gpr_metadata: dict[str, Any] | None,
        warnings: list[str],
    ) -> tuple[np.ndarray, np.ndarray, str | None]:
        if gpr_metadata and gpr_metadata.get("artifact_path"):
            try:
                gpr_artifact_path = self.ensure_gpr_artifact(gpr_metadata, warnings)
                gpr_model = self.load_gpr_model(gpr_artifact_path)
                gpr_features_source = future_features.copy()
                gpr_features_source["xgboost_prediction"] = xgboost_predictions
                feature_columns = _json_list(
                    gpr_metadata.get("feature_columns"),
                    default=GPR_FEATURE_COLUMNS,
                )
                gpr_features, alignment_warnings = self.align_feature_frame(
                    gpr_features_source,
                    feature_columns,
                )
                warnings.extend(f"GPR feature alignment: {item}" for item in alignment_warnings)
                residual_mean, residual_std = gpr_model.predict(
                    gpr_features,
                    return_std=True,
                )
                residual_mean_array = np.asarray(residual_mean, dtype=float)
                residual_std_array = np.maximum(np.asarray(residual_std, dtype=float), 0.0)
                return residual_mean_array, residual_std_array, str(
                    gpr_metadata.get("gpr_run_id")
                )
            except Exception as exc:
                logger.warning("GPR inference failed; using fallback uncertainty: %s", exc)
                warnings.append(
                    "GPR artifact unavailable; uncertainty uses historical residual "
                    f"standard deviation fallback. Reason: {exc}"
                )

        fallback_std = self.get_historical_residual_std_fallback()
        residual_mean = np.zeros(len(xgboost_predictions), dtype=float)
        residual_std = np.full(len(xgboost_predictions), fallback_std, dtype=float)
        return residual_mean, residual_std, None

    def get_historical_residual_std_fallback(self) -> float:
        with self.session_factory() as session:
            residual_std = session.scalar(
                text(
                    """
                    WITH latest AS (
                        SELECT decision_run_id
                        FROM forecast_decision_metrics
                        ORDER BY created_at DESC, id DESC
                        LIMIT 1
                    )
                    SELECT AVG(p.residual_std)
                    FROM forecast_decision_predictions AS p
                    JOIN latest ON latest.decision_run_id = p.decision_run_id
                    WHERE p.residual_std IS NOT NULL
                    """
                )
            )
            if residual_std is not None:
                return max(float(residual_std), 0.0)

            interval_width = session.scalar(
                text(
                    """
                    SELECT COALESCE(mean_interval_width, median_interval_width)
                    FROM forecast_decision_metrics
                    WHERE COALESCE(mean_interval_width, median_interval_width) IS NOT NULL
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                )
            )
            if interval_width is not None:
                return max(float(interval_width) / (2 * 1.96), 0.0)

            xgboost_mae = session.scalar(
                text(
                    """
                    SELECT mae
                    FROM xgboost_metrics
                    WHERE mae IS NOT NULL
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                )
            )
            if xgboost_mae is not None:
                return max(float(xgboost_mae), 0.0)
        return 100.0

    def assign_risk_levels(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = dataframe.copy()
        return assign_risk_levels(frame)

    def store_day_ahead_forecast(
        self,
        forecast_frame: pd.DataFrame,
        forecast_run_id: str,
        target_date: date,
        selected_model: str,
        xgboost_training_run_id: str | None,
        gpr_run_id: str | None,
        decision_run_id: str | None,
        model_version: str,
        generation_method: str,
        warnings: list[str],
        generated_at: datetime,
        session: Session | None = None,
    ) -> int:
        if forecast_frame.empty:
            return 0
        statement = text(
            """
            INSERT INTO day_ahead_forecasts (
                forecast_run_id,
                target_date,
                "timestamp",
                horizon_hour,
                selected_model,
                xgboost_prediction,
                residual_mean,
                residual_std,
                forecast_ptf,
                lower_bound_95,
                upper_bound_95,
                interval_width_95,
                risk_level,
                xgboost_training_run_id,
                gpr_run_id,
                decision_run_id,
                model_version,
                generation_method,
                warnings,
                generated_at
            )
            VALUES (
                :forecast_run_id,
                :target_date,
                :timestamp,
                :horizon_hour,
                :selected_model,
                :xgboost_prediction,
                :residual_mean,
                :residual_std,
                :forecast_ptf,
                :lower_bound_95,
                :upper_bound_95,
                :interval_width_95,
                :risk_level,
                :xgboost_training_run_id,
                :gpr_run_id,
                :decision_run_id,
                :model_version,
                :generation_method,
                CAST(:warnings AS JSONB),
                :generated_at
            )
            ON CONFLICT (forecast_run_id, "timestamp")
            DO UPDATE SET
                selected_model = EXCLUDED.selected_model,
                xgboost_prediction = EXCLUDED.xgboost_prediction,
                residual_mean = EXCLUDED.residual_mean,
                residual_std = EXCLUDED.residual_std,
                forecast_ptf = EXCLUDED.forecast_ptf,
                lower_bound_95 = EXCLUDED.lower_bound_95,
                upper_bound_95 = EXCLUDED.upper_bound_95,
                interval_width_95 = EXCLUDED.interval_width_95,
                risk_level = EXCLUDED.risk_level,
                warnings = EXCLUDED.warnings
            """
        )
        rows = [
            {
                "forecast_run_id": forecast_run_id,
                "target_date": target_date,
                "timestamp": _python_value(row["timestamp"]),
                "horizon_hour": int(row["horizon_hour"]),
                "selected_model": selected_model,
                "xgboost_prediction": _python_value(row["xgboost_prediction"]),
                "residual_mean": _python_value(row["residual_mean"]),
                "residual_std": _python_value(row["residual_std"]),
                "forecast_ptf": _python_value(row["forecast_ptf"]),
                "lower_bound_95": _python_value(row["lower_bound_95"]),
                "upper_bound_95": _python_value(row["upper_bound_95"]),
                "interval_width_95": _python_value(row["interval_width_95"]),
                "risk_level": row["risk_level"],
                "xgboost_training_run_id": xgboost_training_run_id,
                "gpr_run_id": gpr_run_id,
                "decision_run_id": decision_run_id,
                "model_version": model_version,
                "generation_method": generation_method,
                "warnings": json.dumps(warnings),
                "generated_at": generated_at,
            }
            for row in forecast_frame.to_dict(orient="records")
        ]
        owns_session = session is None
        database_session = session or self.session_factory()
        try:
            result = database_session.execute(statement, rows)
            if owns_session:
                database_session.commit()
            return result.rowcount if result.rowcount >= 0 else len(rows)
        except SQLAlchemyError:
            if owns_session:
                database_session.rollback()
            raise
        finally:
            if owns_session:
                database_session.close()

    def run_day_ahead_forecast(
        self,
        target_date: date | None = None,
        horizon_hours: int = DEFAULT_HORIZON_HOURS,
        model_version: str = DEFAULT_MODEL_VERSION,
    ) -> dict[str, Any]:
        if horizon_hours <= 0:
            raise ValueError("horizon_hours must be positive")
        if not model_version.strip():
            raise ValueError("model_version must not be empty")

        forecast_run_id = str(uuid.uuid4())
        generated_at = datetime.now(tz=ISTANBUL_TIMEZONE)
        warnings: list[str] = []
        errors: list[str] = []

        history = self.load_recent_ptf_history()
        if history.empty:
            return self._summary(
                forecast_run_id,
                target_date,
                horizon_hours,
                model_version,
                None,
                None,
                None,
                0,
                None,
                None,
                warnings,
                ["No PTF history found in ptf_hourly"],
            )
        latest_history_local = pd.Timestamp(history["timestamp"].max()).tz_convert(
            ISTANBUL_TIMEZONE
        )
        resolved_target_date = target_date or (
            latest_history_local.date() + timedelta(days=1)
        )

        xgboost_metadata = self.get_latest_xgboost_artifact()
        if xgboost_metadata is None:
            return self._summary(
                forecast_run_id,
                resolved_target_date,
                horizon_hours,
                model_version,
                None,
                None,
                None,
                0,
                None,
                None,
                warnings,
                ["No successful XGBoost model artifact found"],
            )

        future_features, feature_warnings = self.build_future_feature_frame(
            target_date=resolved_target_date,
            horizon_hours=horizon_hours,
            history=history,
        )
        warnings.extend(feature_warnings)

        xgboost_feature_columns = _json_list(xgboost_metadata.get("feature_columns"))
        if not xgboost_feature_columns:
            errors.append("Latest XGBoost metrics row has no feature_columns metadata")
            return self._summary(
                forecast_run_id,
                resolved_target_date,
                horizon_hours,
                model_version,
                str(xgboost_metadata.get("training_run_id")),
                None,
                None,
                0,
                None,
                None,
                warnings,
                errors,
            )

        xgboost_features, alignment_warnings = self.align_feature_frame(
            future_features,
            xgboost_feature_columns,
        )
        warnings.extend(f"XGBoost feature alignment: {item}" for item in alignment_warnings)

        xgboost_artifact_path = self.ensure_xgboost_artifact(
            xgboost_metadata,
            warnings,
        )
        xgboost_model = self.load_xgboost_model(xgboost_artifact_path)
        xgboost_predictions = self.predict_xgboost(xgboost_model, xgboost_features)

        gpr_metadata = self.get_latest_gpr_artifact()
        residual_mean, residual_std, effective_gpr_run_id = self.predict_uncertainty(
            future_features,
            xgboost_predictions,
            gpr_metadata,
            warnings,
        )
        decision_metadata = self.get_latest_decision_metadata()
        decision_run_id = (
            str(decision_metadata["decision_run_id"])
            if decision_metadata and decision_metadata.get("decision_run_id")
            else None
        )
        gpr_run_id = effective_gpr_run_id or (
            str(gpr_metadata["gpr_run_id"])
            if gpr_metadata and gpr_metadata.get("gpr_run_id")
            else None
        )

        forecast_frame = pd.DataFrame(
            {
                "timestamp": future_features["timestamp"],
                "horizon_hour": np.arange(1, horizon_hours + 1),
                "xgboost_prediction": xgboost_predictions,
                "residual_mean": residual_mean,
                "residual_std": residual_std,
            }
        )
        forecast_frame["forecast_ptf"] = forecast_frame["xgboost_prediction"]
        forecast_frame["lower_bound_95"] = (
            forecast_frame["forecast_ptf"] - 1.96 * forecast_frame["residual_std"]
        )
        forecast_frame["upper_bound_95"] = (
            forecast_frame["forecast_ptf"] + 1.96 * forecast_frame["residual_std"]
        )
        forecast_frame["interval_width_95"] = (
            forecast_frame["upper_bound_95"] - forecast_frame["lower_bound_95"]
        )
        forecast_frame = self.assign_risk_levels(forecast_frame)

        rows_generated = self.store_day_ahead_forecast(
            forecast_frame,
            forecast_run_id=forecast_run_id,
            target_date=resolved_target_date,
            selected_model="xgboost",
            xgboost_training_run_id=str(xgboost_metadata["training_run_id"]),
            gpr_run_id=gpr_run_id,
            decision_run_id=decision_run_id,
            model_version=model_version,
            generation_method=GENERATION_METHOD,
            warnings=warnings,
            generated_at=generated_at,
        )
        return self._summary(
            forecast_run_id,
            resolved_target_date,
            horizon_hours,
            model_version,
            str(xgboost_metadata["training_run_id"]),
            gpr_run_id,
            decision_run_id,
            rows_generated,
            _series_min_iso(forecast_frame["timestamp"]),
            _series_max_iso(forecast_frame["timestamp"]),
            warnings,
            errors,
        )

    def get_latest_forecast(self) -> dict[str, Any]:
        with self.session_factory() as session:
            latest_run_id = session.scalar(
                text(
                    """
                    SELECT forecast_run_id
                    FROM day_ahead_forecasts
                    ORDER BY generated_at DESC, id DESC
                    LIMIT 1
                    """
                )
            )
            if latest_run_id is None:
                return {
                    "latest_forecast_run_id": None,
                    "target_date": None,
                    "generated_at": None,
                    "rows": 0,
                    "summary": {},
                    "forecasts": [],
                }
            rows = [
                dict(row)
                for row in session.execute(
                    text(
                        """
                        SELECT *
                        FROM day_ahead_forecasts
                        WHERE forecast_run_id = :forecast_run_id
                        ORDER BY "timestamp"
                        """
                    ),
                    {"forecast_run_id": latest_run_id},
                ).mappings()
            ]
        return _latest_response_from_rows(rows)

    def get_status(self) -> dict[str, Any]:
        with self.session_factory() as session:
            row = session.execute(
                text(
                    """
                    SELECT
                        COUNT(*) AS total_rows,
                        COUNT(DISTINCT forecast_run_id) AS total_runs,
                        COALESCE(
                            ARRAY_AGG(DISTINCT model_version)
                                FILTER (WHERE model_version IS NOT NULL),
                            ARRAY[]::TEXT[]
                        ) AS available_model_versions
                    FROM day_ahead_forecasts
                    """
                )
            ).mappings().one()
            latest = session.execute(
                text(
                    """
                    SELECT forecast_run_id, target_date, generated_at
                    FROM day_ahead_forecasts
                    ORDER BY generated_at DESC, id DESC
                    LIMIT 1
                    """
                )
            ).mappings().one_or_none()
        return {
            "total_rows": int(row["total_rows"] or 0),
            "total_runs": int(row["total_runs"] or 0),
            "latest_forecast_run_id": latest["forecast_run_id"] if latest else None,
            "latest_target_date": latest["target_date"] if latest else None,
            "latest_generated_at": latest["generated_at"] if latest else None,
            "available_model_versions": list(row["available_model_versions"] or []),
        }

    def _summary(
        self,
        forecast_run_id: str,
        target_date: date | None,
        horizon_hours: int,
        model_version: str,
        xgboost_training_run_id: str | None,
        gpr_run_id: str | None,
        decision_run_id: str | None,
        rows_generated: int,
        min_timestamp: str | None,
        max_timestamp: str | None,
        warnings: list[str],
        errors: list[str],
    ) -> dict[str, Any]:
        return {
            "forecast_run_id": forecast_run_id,
            "target_date": target_date.isoformat() if target_date else None,
            "horizon_hours": horizon_hours,
            "selected_model": "xgboost",
            "xgboost_training_run_id": xgboost_training_run_id,
            "gpr_run_id": gpr_run_id,
            "decision_run_id": decision_run_id,
            "generation_method": GENERATION_METHOD,
            "model_version": model_version,
            "rows_generated": rows_generated,
            "min_timestamp": min_timestamp,
            "max_timestamp": max_timestamp,
            "warnings": warnings,
            "errors": errors,
        }


def build_target_timestamps(target_date: date, horizon_hours: int) -> pd.DatetimeIndex:
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")
    start = datetime.combine(target_date, time.min, tzinfo=ISTANBUL_TIMEZONE)
    return pd.date_range(start=start, periods=horizon_hours, freq="h")


def assign_risk_levels(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy()
    if frame.empty:
        frame["risk_level"] = pd.Series(dtype="object")
        return frame
    widths = pd.to_numeric(frame["interval_width_95"], errors="coerce").fillna(0.0)
    q50 = widths.quantile(0.50)
    q85 = widths.quantile(0.85)
    frame["risk_level"] = np.where(
        widths <= q50,
        "LOW",
        np.where(widths <= q85, "MEDIUM", "HIGH"),
    )
    return frame


def _calendar_feature_record(timestamp: pd.Timestamp) -> dict[str, Any]:
    local_timestamp = timestamp.tz_convert(ISTANBUL_TIMEZONE)
    day_of_week = int(local_timestamp.dayofweek)
    hour = int(local_timestamp.hour)
    month = int(local_timestamp.month)
    return {
        "hour": hour,
        "day_of_week": day_of_week,
        "day_of_month": int(local_timestamp.day),
        "day_of_year": int(local_timestamp.dayofyear),
        "week_of_year": int(local_timestamp.isocalendar().week),
        "month": month,
        "quarter": int(local_timestamp.quarter),
        "year": int(local_timestamp.year),
        "is_weekend": day_of_week >= 5,
        "is_month_start": bool(local_timestamp.is_month_start),
        "is_month_end": bool(local_timestamp.is_month_end),
        "is_peak_hour": day_of_week < 5 and 8 <= hour <= 19,
        "is_business_hour": day_of_week < 5 and 9 <= hour <= 17,
        "season": _season_for_month(month),
    }


def _season_for_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def _add_future_rolling_features(
    record: dict[str, Any],
    previous_values: pd.Series,
    used_rolling_fallbacks: set[str],
) -> None:
    for hours, label in [(24, "24h"), (7 * 24, "7d")]:
        values = previous_values.tail(hours)
        if len(values) < hours:
            used_rolling_fallbacks.add(label)
        record[f"ptf_{label}_mean"] = float(values.mean()) if not values.empty else np.nan
        record[f"ptf_{label}_std"] = float(values.std()) if len(values) > 1 else 0.0
        record[f"ptf_{label}_min"] = float(values.min()) if not values.empty else np.nan
        record[f"ptf_{label}_max"] = float(values.max()) if not values.empty else np.nan

    values_30d = previous_values.tail(30 * 24)
    if len(values_30d) < 30 * 24:
        used_rolling_fallbacks.add("30d")
    record["ptf_30d_mean"] = float(values_30d.mean()) if not values_30d.empty else np.nan
    record["ptf_30d_std"] = float(values_30d.std()) if len(values_30d) > 1 else 0.0


def _build_numeric_feature_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    feature_parts: list[pd.DataFrame] = []
    for column in dataframe.columns:
        if column in EXCLUDED_FEATURE_COLUMNS:
            continue
        if column == "season":
            feature_parts.append(
                pd.get_dummies(
                    dataframe[column].fillna("unknown").astype(str),
                    prefix="season",
                    dtype=int,
                )
            )
            continue
        series = dataframe[column]
        if pd.api.types.is_bool_dtype(series):
            feature_parts.append(series.astype(int).to_frame(column))
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            feature_parts.append(numeric.to_frame(column))
    if not feature_parts:
        raise ValueError("No usable future model feature columns were found")
    return pd.concat(feature_parts, axis=1).reindex(
        sorted(pd.concat(feature_parts, axis=1).columns),
        axis=1,
    )


def _latest_response_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "latest_forecast_run_id": None,
            "target_date": None,
            "generated_at": None,
            "rows": 0,
            "summary": {},
            "forecasts": [],
        }
    frame = pd.DataFrame(rows)
    for column in [
        "forecast_ptf",
        "lower_bound_95",
        "upper_bound_95",
        "interval_width_95",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    risk_counts = frame["risk_level"].value_counts().to_dict()
    forecasts = [_json_ready(row) for row in rows]
    return {
        "latest_forecast_run_id": rows[0]["forecast_run_id"],
        "target_date": _date_iso(rows[0]["target_date"]),
        "generated_at": _datetime_iso(rows[0]["generated_at"]),
        "rows": len(rows),
        "summary": {
            "mean_forecast": _safe_float(frame["forecast_ptf"].mean()),
            "min_forecast": _safe_float(frame["forecast_ptf"].min()),
            "max_forecast": _safe_float(frame["forecast_ptf"].max()),
            "mean_interval_width": _safe_float(frame["interval_width_95"].mean()),
            "risk_level_counts": {
                "LOW": int(risk_counts.get("LOW", 0)),
                "MEDIUM": int(risk_counts.get("MEDIUM", 0)),
                "HIGH": int(risk_counts.get("HIGH", 0)),
            },
        },
        "forecasts": forecasts,
    }


def _json_list(value: Any, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return list(default or [])
        if isinstance(decoded, list):
            return [str(item) for item in decoded]
    return list(default or [])


def _mapping_to_dict(value: Any) -> dict[str, Any] | None:
    return dict(value) if value is not None else None


def _finite_median(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return 0.0
    return float(numeric.median())


def _python_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _series_min_iso(series: pd.Series) -> str | None:
    if series.empty:
        return None
    return pd.Timestamp(series.min()).isoformat()


def _series_max_iso(series: pd.Series) -> str | None:
    if series.empty:
        return None
    return pd.Timestamp(series.max()).isoformat()


def _date_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _datetime_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value
