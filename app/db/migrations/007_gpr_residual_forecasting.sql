CREATE TABLE IF NOT EXISTS gpr_residual_predictions (
    id BIGSERIAL PRIMARY KEY,
    "timestamp" TIMESTAMPTZ NOT NULL,
    gpr_run_id TEXT NOT NULL,
    xgboost_training_run_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    xgboost_prediction NUMERIC NOT NULL,
    residual_mean NUMERIC NOT NULL,
    residual_std NUMERIC NOT NULL,
    final_prediction NUMERIC NOT NULL,
    actual NUMERIC NOT NULL,
    lower_bound_95 NUMERIC NOT NULL,
    upper_bound_95 NUMERIC NOT NULL,
    interval_width_95 NUMERIC NOT NULL,
    risk_level TEXT NOT NULL,
    error NUMERIC,
    absolute_error NUMERIC,
    percentage_error NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE ("timestamp", gpr_run_id)
);

CREATE INDEX IF NOT EXISTS ix_gpr_residual_predictions_run_timestamp
    ON gpr_residual_predictions (gpr_run_id, "timestamp");

CREATE INDEX IF NOT EXISTS ix_gpr_residual_predictions_xgboost_run
    ON gpr_residual_predictions (xgboost_training_run_id);

CREATE TABLE IF NOT EXISTS gpr_residual_metrics (
    id BIGSERIAL PRIMARY KEY,
    gpr_run_id TEXT NOT NULL,
    xgboost_training_run_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    residual_train_start TIMESTAMPTZ,
    residual_train_end TIMESTAMPTZ,
    residual_test_start TIMESTAMPTZ,
    residual_test_end TIMESTAMPTZ,
    train_rows INTEGER,
    test_rows INTEGER,
    max_train_rows INTEGER,
    mae NUMERIC,
    rmse NUMERIC,
    mape NUMERIC,
    smape NUMERIC,
    r2 NUMERIC,
    count INTEGER,
    mean_actual NUMERIC,
    mean_prediction NUMERIC,
    max_error NUMERIC,
    median_absolute_error NUMERIC,
    interval_coverage_95 NUMERIC,
    mean_interval_width NUMERIC,
    median_interval_width NUMERIC,
    low_risk_count INTEGER,
    medium_risk_count INTEGER,
    high_risk_count INTEGER,
    xgboost_comparison JSONB,
    baseline_comparison JSONB,
    feature_columns JSONB,
    model_params JSONB,
    artifact_path TEXT,
    warnings JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (gpr_run_id)
);

CREATE INDEX IF NOT EXISTS ix_gpr_residual_metrics_created_at
    ON gpr_residual_metrics (created_at DESC);

CREATE INDEX IF NOT EXISTS ix_gpr_residual_metrics_model_version
    ON gpr_residual_metrics (model_version);
