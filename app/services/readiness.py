import os
from collections.abc import Callable
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal


class SystemReadinessService:
    def __init__(
        self,
        session_factory: Callable[[], Session] = SessionLocal,
    ) -> None:
        self.session_factory = session_factory

    def get_readiness(self) -> dict[str, Any]:
        dashboard_url = f"http://localhost:{os.getenv('STREAMLIT_PORT', '8501')}"
        swagger_url = f"http://localhost:{os.getenv('API_PORT', '8000')}/docs"
        mlflow_url = f"http://localhost:{os.getenv('MLFLOW_PORT', '5000')}"
        base: dict[str, Any] = {
            "api_healthy": True,
            "db_reachable": False,
            "ptf_rows": None,
            "feature_rows": None,
            "latest_forecast_decision_run": None,
            "latest_day_ahead_forecast_run": None,
            "latest_pipeline_run": None,
            "latest_monitoring_status": None,
            "dashboard_url": dashboard_url,
            "swagger_url": swagger_url,
            "mlflow_url": mlflow_url,
            "details": {
                "environment": settings.environment,
                "app_version": settings.app_version,
            },
        }
        try:
            with self.session_factory() as session:
                row = session.execute(
                    text(
                        """
                        SELECT
                            (SELECT COUNT(*) FROM ptf_hourly) AS ptf_rows,
                            (SELECT COUNT(*) FROM features_ptf_hourly) AS feature_rows,
                            (
                                SELECT decision_run_id
                                FROM forecast_decision_metrics
                                ORDER BY created_at DESC, id DESC
                                LIMIT 1
                            ) AS latest_forecast_decision_run,
                            (
                                SELECT forecast_run_id
                                FROM day_ahead_forecasts
                                ORDER BY generated_at DESC, id DESC
                                LIMIT 1
                            ) AS latest_day_ahead_forecast_run,
                            (
                                SELECT pipeline_run_id
                                FROM pipeline_runs
                                ORDER BY started_at DESC, id DESC
                                LIMIT 1
                            ) AS latest_pipeline_run,
                            (
                                SELECT status
                                FROM monitoring_snapshots
                                ORDER BY created_at DESC, id DESC
                                LIMIT 1
                            ) AS latest_monitoring_status
                        """
                    )
                ).mappings().one()
            base.update(
                {
                    "db_reachable": True,
                    "ptf_rows": int(row["ptf_rows"] or 0),
                    "feature_rows": int(row["feature_rows"] or 0),
                    "latest_forecast_decision_run": row[
                        "latest_forecast_decision_run"
                    ],
                    "latest_day_ahead_forecast_run": row[
                        "latest_day_ahead_forecast_run"
                    ],
                    "latest_pipeline_run": row["latest_pipeline_run"],
                    "latest_monitoring_status": row["latest_monitoring_status"],
                }
            )
        except Exception as exc:
            base["details"]["db_error"] = str(exc)
        return base
