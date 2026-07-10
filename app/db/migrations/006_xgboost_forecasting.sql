CREATE TABLE IF NOT EXISTS xgboost_predictions (
    id BIGSERIAL PRIMARY KEY,
    "timestamp" TIMESTAMPTZ NOT NULL,
    model_version TEXT NOT NULL,
    prediction NUMERIC NOT NULL,
    actual NUMERIC NOT NULL,
    error NUMERIC,
    absolute_error NUMERIC,
    percentage_error NUMERIC,
    training_run_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE ("timestamp", model_version, training_run_id)
);

CREATE INDEX IF NOT EXISTS ix_xgboost_predictions_run_timestamp
    ON xgboost_predictions (training_run_id, "timestamp");

CREATE INDEX IF NOT EXISTS ix_xgboost_predictions_model_version
    ON xgboost_predictions (model_version);

CREATE TABLE IF NOT EXISTS xgboost_metrics (
    id BIGSERIAL PRIMARY KEY,
    training_run_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    train_start TIMESTAMPTZ,
    train_end TIMESTAMPTZ,
    test_start TIMESTAMPTZ,
    test_end TIMESTAMPTZ,
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
    baseline_comparison JSONB,
    feature_columns JSONB,
    model_params JSONB,
    artifact_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (training_run_id, model_version)
);

CREATE INDEX IF NOT EXISTS ix_xgboost_metrics_created_at
    ON xgboost_metrics (created_at DESC);

CREATE INDEX IF NOT EXISTS ix_xgboost_metrics_model_version
    ON xgboost_metrics (model_version);
