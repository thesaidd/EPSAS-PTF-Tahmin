import argparse
from pathlib import Path
from pprint import pprint
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.monitoring.ptf_monitoring import PtfMonitoringService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and store a PTF monitoring snapshot."
    )
    parser.add_argument("--max-ptf-age-hours", type=int, default=168)
    parser.add_argument("--expected-forecast-horizon-hours", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = PtfMonitoringService().build_snapshot(
        max_ptf_age_hours=args.max_ptf_age_hours,
        expected_forecast_horizon_hours=args.expected_forecast_horizon_hours,
    )
    pprint(snapshot)


if __name__ == "__main__":
    main()
