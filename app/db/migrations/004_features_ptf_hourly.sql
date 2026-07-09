CREATE TABLE IF NOT EXISTS features_ptf_hourly (
    "timestamp" TIMESTAMPTZ NOT NULL PRIMARY KEY,
    target_ptf NUMERIC NOT NULL,
    feature_version TEXT NOT NULL DEFAULT 'v1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'features_ptf_hourly'
          AND column_name = 'delivery_time'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'features_ptf_hourly'
          AND column_name = 'timestamp'
    ) THEN
        ALTER TABLE features_ptf_hourly
            RENAME COLUMN delivery_time TO "timestamp";
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'features_ptf_hourly'
          AND column_name = 'feature_set_version'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'features_ptf_hourly'
          AND column_name = 'feature_version'
    ) THEN
        ALTER TABLE features_ptf_hourly
            RENAME COLUMN feature_set_version TO feature_version;
    END IF;
END
$$;

ALTER TABLE features_ptf_hourly
    ADD COLUMN IF NOT EXISTS target_ptf NUMERIC,
    ADD COLUMN IF NOT EXISTS hour INTEGER,
    ADD COLUMN IF NOT EXISTS day_of_week INTEGER,
    ADD COLUMN IF NOT EXISTS day_of_month INTEGER,
    ADD COLUMN IF NOT EXISTS day_of_year INTEGER,
    ADD COLUMN IF NOT EXISTS week_of_year INTEGER,
    ADD COLUMN IF NOT EXISTS month INTEGER,
    ADD COLUMN IF NOT EXISTS quarter INTEGER,
    ADD COLUMN IF NOT EXISTS year INTEGER,
    ADD COLUMN IF NOT EXISTS is_weekend BOOLEAN,
    ADD COLUMN IF NOT EXISTS is_month_start BOOLEAN,
    ADD COLUMN IF NOT EXISTS is_month_end BOOLEAN,
    ADD COLUMN IF NOT EXISTS is_peak_hour BOOLEAN,
    ADD COLUMN IF NOT EXISTS is_business_hour BOOLEAN,
    ADD COLUMN IF NOT EXISTS season TEXT,
    ADD COLUMN IF NOT EXISTS ptf_lag_1 NUMERIC,
    ADD COLUMN IF NOT EXISTS ptf_lag_2 NUMERIC,
    ADD COLUMN IF NOT EXISTS ptf_lag_3 NUMERIC,
    ADD COLUMN IF NOT EXISTS ptf_lag_24 NUMERIC,
    ADD COLUMN IF NOT EXISTS ptf_lag_48 NUMERIC,
    ADD COLUMN IF NOT EXISTS ptf_lag_72 NUMERIC,
    ADD COLUMN IF NOT EXISTS ptf_lag_168 NUMERIC,
    ADD COLUMN IF NOT EXISTS ptf_24h_mean NUMERIC,
    ADD COLUMN IF NOT EXISTS ptf_24h_std NUMERIC,
    ADD COLUMN IF NOT EXISTS ptf_24h_min NUMERIC,
    ADD COLUMN IF NOT EXISTS ptf_24h_max NUMERIC,
    ADD COLUMN IF NOT EXISTS ptf_7d_mean NUMERIC,
    ADD COLUMN IF NOT EXISTS ptf_7d_std NUMERIC,
    ADD COLUMN IF NOT EXISTS ptf_7d_min NUMERIC,
    ADD COLUMN IF NOT EXISTS ptf_7d_max NUMERIC,
    ADD COLUMN IF NOT EXISTS ptf_30d_mean NUMERIC,
    ADD COLUMN IF NOT EXISTS ptf_30d_std NUMERIC,
    ADD COLUMN IF NOT EXISTS ptf_diff_1 NUMERIC,
    ADD COLUMN IF NOT EXISTS ptf_diff_24 NUMERIC,
    ADD COLUMN IF NOT EXISTS ptf_pct_change_1 NUMERIC,
    ADD COLUMN IF NOT EXISTS ptf_pct_change_24 NUMERIC,
    ADD COLUMN IF NOT EXISTS feature_version TEXT,
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;

UPDATE features_ptf_hourly AS features
SET target_ptf = source.ptf_tl
FROM ptf_hourly AS source
WHERE features."timestamp" = source."timestamp"
  AND features.target_ptf IS NULL;

UPDATE features_ptf_hourly
SET feature_version = COALESCE(feature_version, 'v1'),
    created_at = COALESCE(created_at, NOW()),
    updated_at = COALESCE(updated_at, created_at, NOW());

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM features_ptf_hourly WHERE target_ptf IS NULL
    ) THEN
        RAISE EXCEPTION
            'Cannot migrate features_ptf_hourly: legacy rows have no matching PTF target';
    END IF;
END
$$;

ALTER TABLE features_ptf_hourly
    ALTER COLUMN target_ptf SET NOT NULL,
    ALTER COLUMN feature_version TYPE TEXT USING feature_version::TEXT,
    ALTER COLUMN feature_version SET DEFAULT 'v1',
    ALTER COLUMN feature_version SET NOT NULL,
    ALTER COLUMN created_at SET DEFAULT NOW(),
    ALTER COLUMN created_at SET NOT NULL,
    ALTER COLUMN updated_at SET DEFAULT NOW(),
    ALTER COLUMN updated_at SET NOT NULL;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'features_ptf_hourly'
          AND column_name = 'features'
    ) THEN
        ALTER TABLE features_ptf_hourly
            ALTER COLUMN features SET DEFAULT '{}'::JSONB;
    END IF;
END
$$;

DO $$
DECLARE
    primary_key_name TEXT;
    primary_key_columns TEXT[];
BEGIN
    SELECT constraint_name
    INTO primary_key_name
    FROM information_schema.table_constraints
    WHERE table_schema = 'public'
      AND table_name = 'features_ptf_hourly'
      AND constraint_type = 'PRIMARY KEY';

    IF primary_key_name IS NOT NULL THEN
        SELECT ARRAY_AGG(column_name ORDER BY ordinal_position)
        INTO primary_key_columns
        FROM information_schema.key_column_usage
        WHERE table_schema = 'public'
          AND table_name = 'features_ptf_hourly'
          AND constraint_name = primary_key_name;
    END IF;

    IF primary_key_columns IS DISTINCT FROM ARRAY['timestamp']::TEXT[] THEN
        IF EXISTS (
            SELECT 1
            FROM features_ptf_hourly
            GROUP BY "timestamp"
            HAVING COUNT(*) > 1
        ) THEN
            RAISE EXCEPTION
                'Cannot use timestamp primary key: duplicate legacy feature timestamps exist';
        END IF;

        IF primary_key_name IS NOT NULL THEN
            EXECUTE format(
                'ALTER TABLE features_ptf_hourly DROP CONSTRAINT %I',
                primary_key_name
            );
        END IF;

        ALTER TABLE features_ptf_hourly
            ADD PRIMARY KEY ("timestamp");
    END IF;
END
$$;

SELECT create_hypertable(
    'features_ptf_hourly',
    'timestamp',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS ix_features_ptf_hourly_version_timestamp
    ON features_ptf_hourly (feature_version, "timestamp" DESC);
