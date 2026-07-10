from datetime import date

import pandas as pd

from dashboard.components import (
    format_metric_value,
    prepare_prediction_table,
    readable_timestamp,
)
from dashboard.data_access import build_decision_predictions_query


def test_dashboard_modules_import() -> None:
    import dashboard.components
    import dashboard.data_access

    assert dashboard.components is not None
    assert dashboard.data_access is not None


def test_format_metric_value_handles_nulls_and_numbers() -> None:
    assert format_metric_value(None) == "—"
    assert format_metric_value(388.9721) == "388.97"
    assert format_metric_value(93.991, suffix="%") == "93.99%"
    assert format_metric_value(4560, decimals=0) == "4,560"


def test_readable_timestamp_converts_to_istanbul_time() -> None:
    assert readable_timestamp("2025-12-31T21:00:00+00:00") == "2026-01-01 00:00"


def test_prepare_prediction_table_rounds_numeric_columns() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2025-12-31T21:00:00+00:00"],
            "selected_model": ["xgboost"],
            "selected_prediction": [2932.1689],
            "actual": [2900.121],
            "absolute_error": [32.0479],
            "lower_bound_95": [1950.4642],
            "upper_bound_95": [3913.8735],
            "risk_level": ["LOW"],
        }
    )

    table = prepare_prediction_table(frame)

    assert table.iloc[0]["timestamp"] == "2026-01-01 00:00"
    assert table.iloc[0]["selected_prediction"] == 2932.17
    assert table.iloc[0]["absolute_error"] == 32.05


def test_prediction_query_builder_adds_optional_filters() -> None:
    query, params = build_decision_predictions_query(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 7),
        risk_levels=["LOW", "HIGH"],
        limit=168,
    )

    assert '"timestamp" >= %(start_date)s' in query
    assert '"timestamp" <= %(end_date)s' in query
    assert "risk_level = ANY(%(risk_levels)s)" in query
    assert "LIMIT %(limit)s" in query
    assert params["risk_levels"] == ["LOW", "HIGH"]
    assert params["limit"] == 168
