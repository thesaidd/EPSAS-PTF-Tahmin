# 10-Minute MVP Demo Script

This script is designed for a local demo of the EPİAŞ PTF Forecasting MVP. It uses existing local data and avoids retraining or live EPİAŞ ingestion unless you explicitly choose to run those flows.

## 0:00-1:00 — Start the stack

```powershell
docker compose up -d --build
docker compose ps
```

Open the three demo surfaces:

- Dashboard: <http://localhost:8501>
- Swagger API: <http://localhost:8000/docs>
- MLflow: <http://localhost:5000>

## 1:00-2:00 — Show system health and readiness

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/api/system/readiness
```

Explain that readiness summarizes API health, database reachability, row counts, latest forecast decision run, latest day-ahead forecast run, latest pipeline run, latest monitoring status, and demo URLs.

## 2:00-3:00 — Show Swagger route groups

Open <http://localhost:8000/docs> and point out:

- EPİAŞ ingestion/client endpoints
- Feature endpoints
- Baseline, XGBoost, GPR, and forecast decision endpoints
- Day-ahead forecast endpoints
- Daily pipeline endpoints
- Monitoring endpoints
- System readiness endpoint

## 3:00-4:00 — Show the dashboard

Open <http://localhost:8501>.

Walk through:

- monitoring and quality summary
- latest daily pipeline status
- day-ahead forecast with confidence interval
- model decision metrics and risk diagnostics

## 4:00-5:00 — Generate or inspect a day-ahead forecast

Safe API call:

```powershell
$body = @{
  horizon_hours = 24
  model_version = "day_ahead_v1"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/api/forecasts/ptf/day-ahead/generate `
  -ContentType "application/json" `
  -Body $body
```

Then inspect:

```powershell
Invoke-RestMethod http://localhost:8000/api/forecasts/ptf/day-ahead/latest
Invoke-RestMethod http://localhost:8000/api/forecasts/ptf/day-ahead/status
```

## 5:00-6:00 — Run the daily pipeline safely

This mode skips live EPİAŞ ingestion and feature rebuilding, so it is demo-friendly.

```powershell
$body = @{
  skip_ingestion = $true
  skip_feature_build = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/api/pipelines/daily-forecast/run `
  -ContentType "application/json" `
  -Body $body
```

Check status:

```powershell
Invoke-RestMethod http://localhost:8000/api/pipelines/daily-forecast/status
```

## 6:00-7:00 — Generate a monitoring snapshot

```powershell
$body = @{
  max_ptf_age_hours = 168
  expected_forecast_horizon_hours = 24
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/api/monitoring/ptf/snapshot `
  -ContentType "application/json" `
  -Body $body
```

Then show:

```powershell
Invoke-RestMethod http://localhost:8000/api/monitoring/ptf/status
Invoke-RestMethod http://localhost:8000/api/monitoring/ptf/latest
```

## 7:00-8:00 — Explain model results

Current validated result from prior training:

- XGBoost MAE: about 255.47
- Baseline MAE: about 416.32
- XGBoost improvement over baseline: about 38.64%
- latest model R²: about 0.849
- 95% interval coverage: about 93.99%
- decision layer selected model: `xgboost`

Explain:

- The baseline establishes a transparent benchmark.
- XGBoost provides the point forecast.
- GPR models residual uncertainty and produces confidence intervals.
- The decision layer keeps the safer product output when GPR correction does not improve point accuracy; in the current latest run it selects XGBoost and keeps GPR uncertainty intervals.

## 8:00-9:00 — Show MLflow and artifacts

Open <http://localhost:5000> and show model runs, metrics, and stored artifacts where available.

## 9:00-10:00 — Business value summary

For an energy company, this MVP demonstrates:

- hourly PTF forecast visibility for day-ahead planning;
- uncertainty-aware risk labeling for volatile hours;
- auditable data/model pipeline history;
- API-first integration path for internal analytics, dashboards, and trading support tools;
- an extendable base for weather, demand, renewables, outages, and market regime features.

## One-command helper

Use the helper for a fast local demo flow:

```powershell
docker compose exec api python scripts/demo_local_mvp.py
docker compose exec api python scripts/demo_local_mvp.py --all
```

The default helper only checks endpoints. `--all` triggers forecast generation, safe pipeline run, and monitoring snapshot creation.
