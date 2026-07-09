from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.api.features import get_ptf_feature_service
from app.main import app
from ml.features.ptf_features import PtfFeatureService

ISTANBUL = ZoneInfo("Europe/Istanbul")


def sample_ptf_frame(periods: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01 00:00",
                periods=periods,
                freq="h",
                tz=ISTANBUL,
            ),
            "ptf_tl": [float(value) for value in range(1, periods + 1)],
        }
    )


def test_calendar_features_use_istanbul_time() -> None:
    service = PtfFeatureService()

    features = service.build_features(sample_ptf_frame(2))

    assert features.loc[0, "hour"] == 0
    assert features.loc[0, "day_of_week"] == 0
    assert features.loc[0, "day_of_month"] == 1
    assert features.loc[0, "month"] == 1
    assert features.loc[0, "quarter"] == 1
    assert features.loc[0, "season"] == "winter"
    assert bool(features.loc[0, "is_month_start"]) is True
    assert bool(features.loc[0, "is_weekend"]) is False


def test_lag_and_change_features_are_chronological() -> None:
    service = PtfFeatureService()

    features = service.build_features(sample_ptf_frame(30))

    assert pd.isna(features.loc[0, "ptf_lag_1"])
    assert features.loc[1, "ptf_lag_1"] == 1.0
    assert features.loc[24, "ptf_lag_24"] == 1.0
    assert features.loc[24, "ptf_diff_24"] == 24.0


def test_rolling_features_exclude_current_target() -> None:
    service = PtfFeatureService()
    dataframe = sample_ptf_frame(25)
    dataframe.loc[24, "ptf_tl"] = 10000.0

    features = service.build_features(dataframe)

    assert features.loc[24, "ptf_24h_mean"] == 12.5
    assert features.loc[24, "ptf_24h_max"] == 24.0


def test_feature_validation_flags_duplicates_and_negative_targets() -> None:
    service = PtfFeatureService()
    features = service.build_features(sample_ptf_frame(3))
    features.loc[1, "timestamp"] = features.loc[0, "timestamp"]
    features.loc[2, "target_ptf"] = -1.0
    features.loc[2, "ptf_pct_change_1"] = np.inf

    validation = service.validate_features(features, expected_rows=3)

    assert any("duplicate" in error.lower() for error in validation.errors)
    assert any("negative" in warning.lower() for warning in validation.warnings)
    assert pd.isna(features.loc[2, "ptf_pct_change_1"])


def test_build_features_preserves_source_row_count() -> None:
    service = PtfFeatureService()
    source = sample_ptf_frame(48)

    features = service.build_features(source)

    assert len(features) == len(source)
    assert features["target_ptf"].tolist() == source["ptf_tl"].tolist()


class FakeFeatureService:
    def get_status(self) -> dict[str, object]:
        return {
            "total_rows": 10,
            "min_timestamp": datetime(2024, 1, 1, tzinfo=ISTANBUL),
            "max_timestamp": datetime(2024, 1, 1, 9, tzinfo=ISTANBUL),
            "latest_updated_at": datetime(2024, 1, 2, tzinfo=ISTANBUL),
            "feature_versions": ["v1"],
        }


def test_feature_routes_are_registered_and_status_works() -> None:
    app.dependency_overrides[get_ptf_feature_service] = lambda: FakeFeatureService()
    try:
        client = TestClient(app)
        response = client.get("/api/features/ptf/status")
        paths = app.openapi()["paths"]
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["total_rows"] == 10
    assert "get" in paths["/api/features/ptf/status"]
    assert "post" in paths["/api/features/ptf/build"]

