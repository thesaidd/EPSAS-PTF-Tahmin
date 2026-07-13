CREATE TABLE IF NOT EXISTS monitoring_snapshots (
    id BIGSERIAL PRIMARY KEY,
    snapshot_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    data_freshness JSONB,
    data_quality JSONB,
    pipeline_health JSONB,
    forecast_health JSONB,
    model_quality JSONB,
    uncertainty_quality JSONB,
    risk_summary JSONB,
    warnings JSONB,
    errors JSONB
);

CREATE INDEX IF NOT EXISTS ix_monitoring_snapshots_created_at
    ON monitoring_snapshots (created_at DESC);

CREATE INDEX IF NOT EXISTS ix_monitoring_snapshots_status
    ON monitoring_snapshots (status);
