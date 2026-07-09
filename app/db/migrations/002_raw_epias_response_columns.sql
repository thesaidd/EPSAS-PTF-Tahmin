DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'raw_epias_responses'
          AND column_name = 'endpoint'
    ) AND NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'raw_epias_responses'
          AND column_name = 'endpoint_name'
    ) THEN
        ALTER TABLE raw_epias_responses
            RENAME COLUMN endpoint TO endpoint_name;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'raw_epias_responses'
          AND column_name = 'requested_at'
    ) AND NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'raw_epias_responses'
          AND column_name = 'fetched_at'
    ) THEN
        ALTER TABLE raw_epias_responses
            RENAME COLUMN requested_at TO fetched_at;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'raw_epias_responses'
          AND column_name = 'response_status'
    ) AND NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'raw_epias_responses'
          AND column_name = 'status_code'
    ) THEN
        ALTER TABLE raw_epias_responses
            RENAME COLUMN response_status TO status_code;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'raw_epias_responses'
          AND column_name = 'response_payload'
    ) AND NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'raw_epias_responses'
          AND column_name = 'response_json'
    ) THEN
        ALTER TABLE raw_epias_responses
            RENAME COLUMN response_payload TO response_json;
    END IF;
END
$$;

ALTER TABLE raw_epias_responses
    ADD COLUMN IF NOT EXISTS endpoint_url TEXT,
    ADD COLUMN IF NOT EXISTS data_start_date DATE,
    ADD COLUMN IF NOT EXISTS data_end_date DATE;

UPDATE raw_epias_responses
SET endpoint_url = endpoint_name
WHERE endpoint_url IS NULL;

UPDATE raw_epias_responses
SET request_payload = '{}'::JSONB
WHERE request_payload IS NULL;

UPDATE raw_epias_responses
SET status_code = 0
WHERE status_code IS NULL;

ALTER TABLE raw_epias_responses
    ALTER COLUMN endpoint_url SET NOT NULL,
    ALTER COLUMN request_payload SET NOT NULL,
    ALTER COLUMN status_code SET NOT NULL;
