CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS raw_epias_responses (
    id BIGSERIAL NOT NULL,
    endpoint_name VARCHAR(255) NOT NULL,
    endpoint_url TEXT NOT NULL,
    request_payload JSONB NOT NULL,
    response_json JSONB NOT NULL,
    status_code INTEGER NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_start_date DATE,
    data_end_date DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, fetched_at),
    CHECK (
        data_start_date IS NULL
        OR data_end_date IS NULL
        OR data_start_date <= data_end_date
    )
);

SELECT create_hypertable(
    'raw_epias_responses',
    'fetched_at',
    if_not_exists => TRUE
);

CREATE TABLE IF NOT EXISTS ptf_hourly (
    "timestamp" TIMESTAMPTZ NOT NULL,
    ptf_tl NUMERIC NOT NULL,
    ptf_usd NUMERIC,
    ptf_eur NUMERIC,
    source TEXT NOT NULL DEFAULT 'epias',
    raw_record JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY ("timestamp")
);

SELECT create_hypertable(
    'ptf_hourly',
    'timestamp',
    if_not_exists => TRUE
);

CREATE TABLE IF NOT EXISTS features_ptf_hourly (
    "timestamp" TIMESTAMPTZ NOT NULL,
    target_ptf NUMERIC NOT NULL,
    hour INTEGER,
    day_of_week INTEGER,
    day_of_month INTEGER,
    day_of_year INTEGER,
    week_of_year INTEGER,
    month INTEGER,
    quarter INTEGER,
    year INTEGER,
    is_weekend BOOLEAN,
    is_month_start BOOLEAN,
    is_month_end BOOLEAN,
    is_peak_hour BOOLEAN,
    is_business_hour BOOLEAN,
    season TEXT,
    ptf_lag_1 NUMERIC,
    ptf_lag_2 NUMERIC,
    ptf_lag_3 NUMERIC,
    ptf_lag_24 NUMERIC,
    ptf_lag_48 NUMERIC,
    ptf_lag_72 NUMERIC,
    ptf_lag_168 NUMERIC,
    ptf_24h_mean NUMERIC,
    ptf_24h_std NUMERIC,
    ptf_24h_min NUMERIC,
    ptf_24h_max NUMERIC,
    ptf_7d_mean NUMERIC,
    ptf_7d_std NUMERIC,
    ptf_7d_min NUMERIC,
    ptf_7d_max NUMERIC,
    ptf_30d_mean NUMERIC,
    ptf_30d_std NUMERIC,
    ptf_diff_1 NUMERIC,
    ptf_diff_24 NUMERIC,
    ptf_pct_change_1 NUMERIC,
    ptf_pct_change_24 NUMERIC,
    feature_version TEXT NOT NULL DEFAULT 'v1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY ("timestamp")
);

SELECT create_hypertable(
    'features_ptf_hourly',
    'timestamp',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS ix_features_ptf_hourly_version_timestamp
    ON features_ptf_hourly (feature_version, "timestamp" DESC);

CREATE TABLE IF NOT EXISTS model_predictions (
    forecast_time TIMESTAMPTZ NOT NULL,
    delivery_time TIMESTAMPTZ NOT NULL,
    model_name VARCHAR(128) NOT NULL,
    model_version VARCHAR(128) NOT NULL,
    point_forecast DOUBLE PRECISION NOT NULL,
    lower_bound DOUBLE PRECISION,
    upper_bound DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (
        forecast_time,
        delivery_time,
        model_name,
        model_version
    ),
    CHECK (lower_bound IS NULL OR upper_bound IS NULL OR lower_bound <= upper_bound)
);

SELECT create_hypertable(
    'model_predictions',
    'delivery_time',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS ix_model_predictions_lookup
    ON model_predictions (model_name, model_version, delivery_time DESC);

CREATE TABLE IF NOT EXISTS model_metrics (
    evaluation_time TIMESTAMPTZ NOT NULL,
    model_name VARCHAR(128) NOT NULL,
    model_version VARCHAR(128) NOT NULL,
    metric_name VARCHAR(128) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    evaluation_window_start TIMESTAMPTZ,
    evaluation_window_end TIMESTAMPTZ,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (
        evaluation_time,
        model_name,
        model_version,
        metric_name
    )
);

SELECT create_hypertable(
    'model_metrics',
    'evaluation_time',
    if_not_exists => TRUE
);
