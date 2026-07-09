# EPİAŞ PTF Forecasting MVP

A production-oriented starting point for a B2B energy forecasting service for
Turkey's Day-Ahead Market. The target is an hourly 24-hour-ahead forecast of the
Market Clearing Price (PTF/MCP) using data from the EPİAŞ Transparency Platform.

This repository currently contains infrastructure and application scaffolding
only. EPİAŞ ingestion, feature engineering, XGBoost point forecasts, and
Gaussian Process residual uncertainty modeling are intentionally not yet
implemented.

## Architecture

The local stack contains:

- **FastAPI** for health checks and, later, forecast serving.
- **Streamlit** for the MVP user interface.
- **PostgreSQL with TimescaleDB** for raw, feature, prediction, and metric data.
- **MLflow** for experiment tracking and model artifacts.
- **Docker Compose** for reproducible local and cloud-oriented deployment.

The Python packages are separated by responsibility:

```text
app/             API, configuration, schemas, services, and database code
data_pipeline/   EPİAŞ adapters, ingestion jobs, and data validation
ml/              Features, models, training, inference, and evaluation
dashboard/       Streamlit MVP
scripts/         Operational and developer scripts
tests/           Automated tests
docs/            Project documentation
```

## Run with Docker Compose

Requirements:

- Docker Engine or Docker Desktop
- Docker Compose v2

Start the complete stack:

```bash
cp .env.example .env
docker compose up --build
```

On Windows PowerShell, use:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

The services are then available at:

- API: <http://localhost:8000>
- API documentation: <http://localhost:8000/docs>
- API health: <http://localhost:8000/health>
- Streamlit: <http://localhost:8501>
- MLflow: <http://localhost:5000>
- PostgreSQL: `localhost:5432`

Stop the stack with:

```bash
docker compose down
```

Use `docker compose down -v` only when you intentionally want to delete local
database and MLflow volumes.

## Environment variables

Copy `.env.example` to `.env` and adjust values as needed.

| Variable | Purpose | Default |
| --- | --- | --- |
| `PROJECT_NAME` | API/project display name | `EPİAŞ PTF Forecasting MVP` |
| `APP_VERSION` | Application version returned by the API | `0.1.0` |
| `ENVIRONMENT` | Runtime environment label | `development` |
| `POSTGRES_USER` | PostgreSQL user | `pepias` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `pepias` |
| `POSTGRES_DB` | PostgreSQL database | `pepias` |
| `DATABASE_URL` | SQLAlchemy-compatible database URL | Compose database URL |
| `POSTGRES_PORT` | Host PostgreSQL port | `5432` |
| `API_PORT` | Host FastAPI port | `8000` |
| `STREAMLIT_PORT` | Host Streamlit port | `8501` |
| `MLFLOW_PORT` | Host MLflow port | `5000` |
| `API_URL` | Internal API URL used by Streamlit | `http://api:8000` |
| `MLFLOW_TRACKING_URI` | MLflow tracking URI used by application code | `http://mlflow:5000` |
| `MLFLOW_BACKEND_STORE_URI` | PostgreSQL URI used for MLflow metadata | Compose database URL |

Do not use the example credentials in a shared or production environment.

## Database initialization

On the first database startup,
`app/db/migrations/001_initial_schema.sql` enables TimescaleDB, creates the
initial tables, and converts time-indexed tables to hypertables. Docker's
PostgreSQL initialization mechanism runs this file only when the database
volume is empty.

## Run tests locally

With Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

PowerShell activation:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

## Development roadmap

1. Implement authenticated EPİAŞ Transparency Platform ingestion and raw
   response persistence.
2. Add schema validation, data-quality checks, retries, and scheduled jobs.
3. Build leakage-safe hourly features and reproducible training datasets.
4. Train and track an XGBoost point-forecasting baseline in MLflow.
5. Model residual uncertainty with Gaussian Process Regression.
6. Add versioned forecast endpoints, model loading, and prediction persistence.
7. Expand the dashboard with forecast curves, confidence intervals, and model
   monitoring.
8. Add CI, automated migrations, observability, secrets management, and
   production deployment configuration.
