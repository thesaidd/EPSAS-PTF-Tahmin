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
| `EPIAS_BASE_URL` | EPİAŞ Transparency Platform base URL | `https://seffaflik.epias.com.tr` |
| `EPIAS_AUTH_URL` | EPİAŞ authentication service base URL | `https://giris.epias.com.tr` |
| `EPIAS_USERNAME` | EPİAŞ account username | Empty |
| `EPIAS_PASSWORD` | EPİAŞ account password | Empty |
| `EPIAS_REQUEST_TIMEOUT` | HTTP timeout in seconds | `30` |
| `EPIAS_MAX_RETRIES` | Retries after a failed HTTP attempt | `3` |

Do not use the example credentials in a shared or production environment.

## EPİAŞ API Client

`data_pipeline/epias/client.py` provides a generic synchronous HTTP client for
EPİAŞ POST endpoints. It sends JSON requests, applies timeouts and bounded
exponential-backoff retries, and raises clear errors for authentication,
transport, HTTP, and JSON parsing failures.
[EPİAŞ's official technical documentation](https://seffaflik.epias.com.tr/electricity-service/technical/tr/index.html)
defines the authentication and `TGT` header contract used here.

Authenticated calls obtain a TGT from
`EPIAS_AUTH_URL/cas/v1/tickets` using form-encoded credentials. The token is
cached in memory, refreshed before its two-hour lifetime expires, and refreshed
once if an API request returns HTTP 401 or 403. Missing credentials do not stop
the API from starting; only a request with `use_auth=true` requires them.
Credentials and TGT values are never included in application logs or health
responses.

Check client configuration without making an external request:

```bash
curl http://localhost:8000/api/epias/health
```

The development-only manual POST route accepts a relative EPİAŞ endpoint:

```bash
curl -X POST http://localhost:8000/api/epias/test-post \
  -H "Content-Type: application/json" \
  -d '{"endpoint":"/electricity-service/v1/example","payload":{},"use_auth":true}'
```

Replace the example endpoint with a valid EPİAŞ POST endpoint and its required
payload. Successful responses are stored in `raw_epias_responses` with the
endpoint identity and URL, request JSON, response JSON, HTTP status, fetch
timestamp, and optional data date range.

For an existing Docker database volume created before Sprint 2, apply the
idempotent compatibility migration once:

```bash
docker compose exec db psql -U pepias -d pepias \
  -f /docker-entrypoint-initdb.d/002_raw_epias_response_columns.sql
```

Fresh database volumes run both initialization files automatically. Full
historical PTF ingestion is intentionally deferred to Sprint 3.

## Database initialization

On the first database startup,
the SQL files under `app/db/migrations` enable TimescaleDB, create the initial
tables, and convert time-indexed tables to hypertables. Docker's PostgreSQL
initialization mechanism runs these files only when the database volume is
empty.

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

1. Implement historical PTF ingestion using the EPİAŞ client.
2. Add schema validation, data-quality checks, and scheduled jobs.
3. Build leakage-safe hourly features and reproducible training datasets.
4. Train and track an XGBoost point-forecasting baseline in MLflow.
5. Model residual uncertainty with Gaussian Process Regression.
6. Add versioned forecast endpoints, model loading, and prediction persistence.
7. Expand the dashboard with forecast curves, confidence intervals, and model
   monitoring.
8. Add CI, automated migrations, observability, secrets management, and
   production deployment configuration.
