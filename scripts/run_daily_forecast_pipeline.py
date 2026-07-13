import argparse
from datetime import date
from pathlib import Path
from pprint import pprint
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipelines.daily_forecast_pipeline import DailyForecastPipelineService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the daily PTF forecast operational pipeline."
    )
    parser.add_argument("--target-date", type=date.fromisoformat, default=None)
    parser.add_argument("--ingest-start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--ingest-end-date", type=date.fromisoformat, default=None)
    parser.add_argument("--skip-ingestion", action="store_true")
    parser.add_argument("--skip-feature-build", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = DailyForecastPipelineService().run_pipeline(
        target_date=args.target_date,
        ingest_start_date=args.ingest_start_date,
        ingest_end_date=args.ingest_end_date,
        skip_ingestion=args.skip_ingestion,
        skip_feature_build=args.skip_feature_build,
    )
    pprint(summary)


if __name__ == "__main__":
    main()
