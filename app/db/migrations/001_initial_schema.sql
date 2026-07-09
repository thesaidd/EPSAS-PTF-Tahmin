CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS raw_epias_responses (
    id BIGSERIAL NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    response_status INTEGER,
    request_payload JSONB,
    response_payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, requested_at)
);

SELECT create_hypertable(
    'raw_epias_responses',
    'requested_at',
    if_not_exists => TRUE
);

CREATE TABLE IF NOT EXISTS ptf_hourly (
    delivery_time TIMESTAMPTZ NOT NULL,
    price_try_mwh DOUBLE PRECISION NOT NULL,
    price_eur_mwh DOUBLE PRECISION,
    price_usd_mwh DOUBLE PRECISION,
    source_updated_at TIMESTAMPTZ,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (delivery_time)
);

SELECT create_hypertable(
    'ptf_hourly',
    'delivery_time',
    if_not_exists => TRUE
);

CREATE TABLE IF NOT EXISTS features_ptf_hourly (
    delivery_time TIMESTAMPTZ NOT NULL,
    feature_set_version VARCHAR(64) NOT NULL,
    features JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (delivery_time, feature_set_version)
);

SELECT create_hypertable(
    'features_ptf_hourly',
    'delivery_time',
    if_not_exists => TRUE
);

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

