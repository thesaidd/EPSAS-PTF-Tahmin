import argparse
from datetime import date
from pathlib import Path
from pprint import pprint
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.day_ahead_ptf import DayAheadPtfForecastService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 24-hour day-ahead PTF forecast."
    )
    parser.add_argument(
        "--target-date",
        type=date.fromisoformat,
        default=None,
        help="Target delivery date as YYYY-MM-DD. Defaults to day after latest PTF history.",
    )
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=24,
        help="Number of forward hourly forecasts to generate.",
    )
    parser.add_argument(
        "--model-version",
        default="day_ahead_v1",
        help="Forecast output model/version label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = DayAheadPtfForecastService().run_day_ahead_forecast(
        target_date=args.target_date,
        horizon_hours=args.horizon_hours,
        model_version=args.model_version,
    )
    pprint(summary)


if __name__ == "__main__":
    main()
