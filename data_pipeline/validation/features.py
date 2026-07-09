from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FeatureValidationResult:
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def validate_ptf_feature_frame(
    dataframe: pd.DataFrame,
    expected_rows: int | None = None,
) -> FeatureValidationResult:
    result = FeatureValidationResult()
    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)

    required_columns = {"timestamp", "target_ptf"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        result.errors.append(
            f"Missing required feature columns: {', '.join(sorted(missing_columns))}"
        )
        return result

    if dataframe["timestamp"].isna().any():
        result.errors.append("Feature timestamps contain null values")
    if dataframe["timestamp"].duplicated().any():
        result.errors.append("Feature timestamps contain duplicate values")
    if dataframe["target_ptf"].isna().any():
        result.errors.append("target_ptf contains null values")
    if expected_rows is not None and len(dataframe) != expected_rows:
        result.errors.append(
            f"Feature row count {len(dataframe)} does not match source row count "
            f"{expected_rows}"
        )

    negative_count = int((dataframe["target_ptf"] < 0).sum())
    if negative_count:
        result.warnings.append(
            f"Found {negative_count} negative PTF target value(s)"
        )

    timestamp_dtype = dataframe["timestamp"].dtype
    if not isinstance(timestamp_dtype, pd.DatetimeTZDtype):
        result.errors.append("Feature timestamps must be timezone-aware")

    return result

