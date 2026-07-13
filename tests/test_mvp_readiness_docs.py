from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.api.router import get_system_readiness_service
from app.main import app
from scripts.demo_local_mvp import build_demo_steps


class FakeReadinessService:
    def get_readiness(self) -> dict[str, Any]:
        return {
            "api_healthy": True,
            "db_reachable": True,
            "ptf_rows": 100,
            "feature_rows": 90,
            "latest_forecast_decision_run": "decision-run",
            "latest_day_ahead_forecast_run": "forecast-run",
            "latest_pipeline_run": "pipeline-run",
            "latest_monitoring_status": "HEALTHY",
            "dashboard_url": "http://localhost:8501",
            "swagger_url": "http://localhost:8000/docs",
            "mlflow_url": "http://localhost:5000",
            "details": {"environment": "test"},
        }


def test_system_readiness_route_shape() -> None:
    app.dependency_overrides[get_system_readiness_service] = FakeReadinessService
    try:
        response = TestClient(app).get("/api/system/readiness")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_healthy"] is True
    assert payload["db_reachable"] is True
    assert payload["ptf_rows"] == 100
    assert payload["feature_rows"] == 90
    assert payload["latest_monitoring_status"] == "HEALTHY"
    assert payload["dashboard_url"] == "http://localhost:8501"
    assert payload["swagger_url"] == "http://localhost:8000/docs"
    assert payload["mlflow_url"] == "http://localhost:5000"


def test_sprint_12_routes_are_registered() -> None:
    paths = {route.path for route in app.routes}

    assert "/api/system/readiness" in paths
    assert "/api/forecasts/ptf/day-ahead/generate" in paths
    assert "/api/pipelines/daily-forecast/run" in paths
    assert "/api/monitoring/ptf/snapshot" in paths


def test_demo_helper_default_steps_are_safe() -> None:
    steps = build_demo_steps()

    assert all(step.method == "GET" for step in steps)
    assert {step.endpoint for step in steps} >= {
        "/health",
        "/api/system/readiness",
        "/api/epias/health",
    }
    assert "/api/epias/ptf/ingest" not in {step.endpoint for step in steps}


def test_demo_helper_all_steps_include_optional_safe_actions() -> None:
    steps = build_demo_steps(
        run_forecast=True,
        run_pipeline=True,
        run_monitoring=True,
    )
    by_endpoint = {step.endpoint: step for step in steps}

    assert by_endpoint["/api/forecasts/ptf/day-ahead/generate"].method == "POST"
    assert by_endpoint["/api/pipelines/daily-forecast/run"].payload == {
        "skip_ingestion": True,
        "skip_feature_build": True,
    }
    assert by_endpoint["/api/monitoring/ptf/snapshot"].payload == {
        "max_ptf_age_hours": 168,
        "expected_forecast_horizon_hours": 24,
    }


def test_final_mvp_docs_exist_and_cover_demo_topics() -> None:
    expected_docs = {
        "README.md": [
            "Final Validation Checklist",
            "GET /api/system/readiness".replace("GET ", ""),
            "Known limitations",
        ],
        "DEMO.md": ["10-Minute MVP Demo Script", "Business value"],
        "ARCHITECTURE.md": ["Database tables", "Future production architecture"],
        "API_EXAMPLES.md": ["/api/monitoring/ptf/status", "PowerShell"],
        "KULLANIM_REHBERI.md": ["Proje ne yapar?", "Dashboard nasıl okunur?"],
        "DEMO_TR.md": ["10 Dakikalık Türkçe Demo", "Problem tanımı"],
    }

    for filename, required_texts in expected_docs.items():
        text = Path(filename).read_text(encoding="utf-8")
        for required_text in required_texts:
            assert required_text in text


def test_dashboard_source_contains_turkish_business_sections() -> None:
    dashboard_source = Path("dashboard/streamlit_app.py").read_text(encoding="utf-8")

    assert "Sistem Nasıl Çalışıyor?" in dashboard_source
    assert "Gün Öncesi PTF Tahmini" in dashboard_source
    assert "Model Karar Katmanı" in dashboard_source
    assert "İzleme ve Kalite Kontrol" in dashboard_source


def test_dockerignore_keeps_local_demo_artifacts_out_of_images() -> None:
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")

    assert ".env" in dockerignore
    assert "artifacts" in dockerignore
    assert ".pytest_cache" in dockerignore
