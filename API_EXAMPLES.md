# API Examples

Copy-paste-ready PowerShell examples for the local MVP stack.

Base URL:

```powershell
$BaseUrl = "http://localhost:8000"
```

## Health

```powershell
Invoke-RestMethod "$BaseUrl/health"
Invoke-RestMethod "$BaseUrl/version"
Invoke-RestMethod "$BaseUrl/api/system/readiness"
```

## EPİAŞ

```powershell
Invoke-RestMethod "$BaseUrl/api/epias/health"
Invoke-RestMethod "$BaseUrl/api/epias/ptf/status"
```

Development-only manual EPİAŞ POST test:

```powershell
$body = @{
  endpoint = "/electricity-service/v1/markets/dam/data/mcp"
  payload = @{
    startDate = "2024-01-01T00:00:00+03:00"
    endDate = "2024-01-01T23:59:59+03:00"
  }
  use_auth = $true
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri "$BaseUrl/api/epias/test-post" `
  -ContentType "application/json" `
  -Body $body
```

## PTF ingestion

```powershell
$body = @{
  start_date = "2024-01-01"
  end_date = "2024-01-31"
  chunk_days = 30
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "$BaseUrl/api/epias/ptf/ingest" `
  -ContentType "application/json" `
  -Body $body
```

## Features

```powershell
Invoke-RestMethod "$BaseUrl/api/features/ptf/status"
```

```powershell
$body = @{
  start_date = "2024-01-01"
  end_date = "2024-01-31"
  feature_version = "v1"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "$BaseUrl/api/features/ptf/build" `
  -ContentType "application/json" `
  -Body $body
```

## Baseline

```powershell
Invoke-RestMethod "$BaseUrl/api/models/baseline/ptf/status"
```

```powershell
$body = @{
  start_date = "2024-01-01"
  end_date = "2024-01-31"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "$BaseUrl/api/models/baseline/ptf/run" `
  -ContentType "application/json" `
  -Body $body
```

## XGBoost

```powershell
Invoke-RestMethod "$BaseUrl/api/models/xgboost/ptf/status"
```

```powershell
$body = @{
  train_start = "2024-01-01"
  train_end = "2024-12-31"
  test_start = "2025-01-01"
  test_end = "2025-01-31"
  model_version = "xgboost_v1"
  feature_version = "v1"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "$BaseUrl/api/models/xgboost/ptf/train" `
  -ContentType "application/json" `
  -Body $body
```

## GPR residual uncertainty

```powershell
Invoke-RestMethod "$BaseUrl/api/models/gpr-residual/ptf/status"
```

```powershell
$body = @{
  xgboost_training_run_id = "replace-with-xgboost-run-id"
  residual_train_start = "2025-01-01"
  residual_train_end = "2025-01-15"
  residual_test_start = "2025-01-16"
  residual_test_end = "2025-01-31"
  model_version = "gpr_residual_v1"
  max_train_rows = 1000
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "$BaseUrl/api/models/gpr-residual/ptf/train" `
  -ContentType "application/json" `
  -Body $body
```

## Forecast decision layer

```powershell
Invoke-RestMethod "$BaseUrl/api/models/forecast-decision/ptf/status"
```

```powershell
$body = @{
  gpr_run_id = "replace-with-gpr-run-id"
  model_version = "decision_v1"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "$BaseUrl/api/models/forecast-decision/ptf/run" `
  -ContentType "application/json" `
  -Body $body
```

## Day-ahead forecast

```powershell
Invoke-RestMethod "$BaseUrl/api/forecasts/ptf/day-ahead/status"
Invoke-RestMethod "$BaseUrl/api/forecasts/ptf/day-ahead/latest"
```

```powershell
$body = @{
  horizon_hours = 24
  model_version = "day_ahead_v1"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "$BaseUrl/api/forecasts/ptf/day-ahead/generate" `
  -ContentType "application/json" `
  -Body $body
```

## Daily pipeline

```powershell
Invoke-RestMethod "$BaseUrl/api/pipelines/daily-forecast/status"
Invoke-RestMethod "$BaseUrl/api/pipelines/daily-forecast/runs"
```

Safe local run:

```powershell
$body = @{
  skip_ingestion = $true
  skip_feature_build = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "$BaseUrl/api/pipelines/daily-forecast/run" `
  -ContentType "application/json" `
  -Body $body
```

## Monitoring

```powershell
Invoke-RestMethod "$BaseUrl/api/monitoring/ptf/status"
Invoke-RestMethod "$BaseUrl/api/monitoring/ptf/latest"
Invoke-RestMethod "$BaseUrl/api/monitoring/ptf/snapshots"
```

```powershell
$body = @{
  max_ptf_age_hours = 168
  expected_forecast_horizon_hours = 24
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "$BaseUrl/api/monitoring/ptf/snapshot" `
  -ContentType "application/json" `
  -Body $body
```
