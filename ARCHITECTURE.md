# Architecture

This MVP is a local/cloud-portable Docker Compose stack for hourly Turkish Day-Ahead Market PTF/MCP forecasting.

## System modules

```text
FastAPI backend
├─ EPİAŞ client and ingestion routes
├─ feature build routes
├─ baseline / XGBoost / GPR / forecast decision routes
├─ day-ahead forecast routes
├─ daily pipeline routes
├─ monitoring routes
└─ system readiness route

Data pipeline
├─ EPİAŞ HTTP client
├─ raw response persistence
├─ PTF historical ingestion
└─ time-series validation

ML package
├─ feature engineering
├─ baseline evaluation
├─ XGBoost training/inference
├─ GPR residual uncertainty modeling
├─ forecast decision layer
├─ day-ahead inference
├─ daily pipeline orchestration
└─ monitoring snapshots

Dashboard
└─ Streamlit data product UI
```

## Database tables

| Table | Purpose |
| --- | --- |
| `raw_epias_responses` | Stores raw EPİAŞ API responses and request metadata for auditability. |
| `ptf_hourly` | Normalized hourly PTF/MCP observations. |
| `features_ptf_hourly` | Leakage-aware feature rows for hourly forecasting. |
| `model_predictions` | Generic prediction storage from early model iterations. |
| `model_metrics` | Generic metric storage from early model iterations. |
| `baseline_metrics` | Baseline forecast evaluation metrics. |
| `xgboost_metrics` | XGBoost training metrics and artifact paths. |
| `xgboost_predictions` | XGBoost test predictions. |
| `gpr_residual_metrics` | GPR residual uncertainty metrics and artifact paths. |
| `gpr_residual_predictions` | GPR-corrected predictions and uncertainty intervals. |
| `forecast_decision_metrics` | Decision-layer evaluation and selected model metadata. |
| `forecast_decision_predictions` | Product-safe selected predictions, intervals, and risk levels. |
| `day_ahead_forecasts` | Latest generated 24-hour day-ahead forecast outputs. |
| `pipeline_runs` | Persisted daily pipeline step status and metadata. |
| `monitoring_snapshots` | Data quality, model quality, forecast health, uncertainty, and risk health snapshots. |

## API route groups

- `/health`, `/version`, `/api/system/readiness`
- `/api/epias/*`
- `/api/features/*`
- `/api/models/baseline/*`
- `/api/models/xgboost/*`
- `/api/models/gpr-residual/*`
- `/api/models/forecast-decision/*`
- `/api/forecasts/ptf/day-ahead/*`
- `/api/pipelines/daily-forecast/*`
- `/api/monitoring/ptf/*`

## ML pipeline stages

1. Ingest hourly PTF data from EPİAŞ.
2. Normalize and upsert to `ptf_hourly`.
3. Build leakage-aware hourly features into `features_ptf_hourly`.
4. Evaluate simple baselines.
5. Train XGBoost point forecasting model.
6. Train GPR residual model for uncertainty intervals.
7. Run the forecast decision layer.
8. Generate a 24-hour day-ahead output.
9. Persist pipeline and monitoring summaries.
10. Present the result through API and Streamlit.

## Artifact handling

Model artifacts are stored under the local `artifacts/` directory and ignored by git. XGBoost is saved using native model serialization. MLflow logging is best-effort: training should not fail just because the tracking server or artifact logging is temporarily unavailable.

## MLflow usage

MLflow runs in its own Docker service and uses a dedicated PostgreSQL database separate from the application database. It is intended for local experiment inspection, metric comparison, and artifact browsing.

Open: <http://localhost:5000>

## Dashboard structure

The Streamlit app contains:

- monitoring and data quality cards;
- latest daily pipeline status;
- latest day-ahead forecast chart/table;
- forecast decision metrics;
- prediction/risk diagnostics;
- links to Swagger, MLflow, and readiness.

Open: <http://localhost:8501>

## Scheduler profile

The optional daily scheduler is isolated behind a Docker Compose profile:

```powershell
docker compose --profile scheduler up -d daily-pipeline-scheduler
```

It is not enabled by default. For local demos, prefer manual API or CLI pipeline runs.

## Data limitations

- The MVP focuses on PTF/MCP history and derived lag/calendar features.
- Live EPİAŞ ingestion requires credentials.
- Day-ahead feature construction does not recursively feed predicted future values into future lag features.
- Exogenous drivers such as demand, renewable generation, outages, weather, FX, and fuel prices are future roadmap items.

## Future production architecture

A production version would likely add:

- managed PostgreSQL/TimescaleDB;
- object storage for model artifacts;
- managed secret storage;
- CI/CD and database migration orchestration;
- model registry promotion gates;
- scheduled retraining/backtesting;
- Prometheus/Grafana or cloud-native observability;
- authentication, RBAC, and tenant isolation;
- queue-based ingestion and pipeline execution;
- richer external data sources.
