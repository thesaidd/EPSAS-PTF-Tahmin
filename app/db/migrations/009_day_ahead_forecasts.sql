CREATE TABLE IF NOT EXISTS day_ahead_forecasts (
    id BIGSERIAL PRIMARY KEY,
    forecast_run_id TEXT NOT NULL,
    target_date DATE NOT NULL,
    "timestamp" TIMESTAMPTZ NOT NULL,
    horizon_hour INTEGER NOT NULL,
    selected_model TEXT NOT NULL,
    xgboost_prediction NUMERIC NOT NULL,
    residual_mean NUMERIC,
    residual_std NUMERIC,
    forecast_ptf NUMERIC NOT NULL,
    lower_bound_95 NUMERIC,
    upper_bound_95 NUMERIC,
    interval_width_95 NUMERIC,
    risk_level TEXT,
    xgboost_training_run_id TEXT,
    gpr_run_id TEXT,
    decision_run_id TEXT,
    model_version TEXT NOT NULL,
    generation_method TEXT NOT NULL,
    warnings JSONB,
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (forecast_run_id, "timestamp"),
    UNIQUE (target_date, horizon_hour, model_version, generated_at)
);

CREATE INDEX IF NOT EXISTS ix_day_ahead_forecasts_target_date
    ON day_ahead_forecasts (target_date);

CREATE INDEX IF NOT EXISTS ix_day_ahead_forecasts_run_id
    ON day_ahead_forecasts (forecast_run_id);

CREATE INDEX IF NOT EXISTS ix_day_ahead_forecasts_timestamp
    ON day_ahead_forecasts ("timestamp");

CREATE INDEX IF NOT EXISTS ix_day_ahead_forecasts_generated_at
    ON day_ahead_forecasts (generated_at DESC);
