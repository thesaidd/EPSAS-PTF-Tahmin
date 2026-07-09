import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.features.ptf_features import PtfFeatureService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ML-ready hourly features from stored PTF data."
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        help="Inclusive start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        help="Inclusive end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--feature-version",
        default="v1",
        help="Feature definition version stored with each row (default: v1).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    service = PtfFeatureService()
    try:
        summary = service.build_and_store_features(
            start_date=args.start_date,
            end_date=args.end_date,
            feature_version=args.feature_version,
        )
    except ValueError as exc:
        print(f"Feature build could not start: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

