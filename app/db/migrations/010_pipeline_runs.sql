CREATE TABLE IF NOT EXISTS pipeline_runs (
    id BIGSERIAL PRIMARY KEY,
    pipeline_run_id TEXT NOT NULL UNIQUE,
    pipeline_name TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    target_date DATE,
    ingest_start_date DATE,
    ingest_end_date DATE,
    forecast_run_id TEXT,
    steps JSONB,
    warnings JSONB,
    errors JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_pipeline_runs_pipeline_name_started_at
    ON pipeline_runs (pipeline_name, started_at DESC);

CREATE INDEX IF NOT EXISTS ix_pipeline_runs_status
    ON pipeline_runs (status);

CREATE INDEX IF NOT EXISTS ix_pipeline_runs_target_date
    ON pipeline_runs (target_date);
