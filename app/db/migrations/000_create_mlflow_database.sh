#!/bin/sh
set -eu

: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${POSTGRES_DB:?POSTGRES_DB is required}"
: "${MLFLOW_DATABASE:=mlflow}"

psql \
    --set=ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    --set=mlflow_database="$MLFLOW_DATABASE" <<'EOSQL'
SELECT format('CREATE DATABASE %I', :'mlflow_database')
WHERE NOT EXISTS (
    SELECT 1
    FROM pg_database
    WHERE datname = :'mlflow_database'
)
\gexec
EOSQL
