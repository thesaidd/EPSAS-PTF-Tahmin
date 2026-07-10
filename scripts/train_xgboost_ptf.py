import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.xgboost_ptf import XGBoostPtfService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the XGBoost PTF model.")
    parser.add_argument("--train-start", type=date.fromisoformat, default=None)
    parser.add_argument("--train-end", type=date.fromisoformat, default=None)
    parser.add_argument("--test-start", type=date.fromisoformat, default=None)
    parser.add_argument("--test-end", type=date.fromisoformat, default=None)
    parser.add_argument("--model-version", default="xgboost_v1")
    parser.add_argument("--feature-version", default="v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary: dict[str, Any] = XGBoostPtfService().run_training(
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        model_version=args.model_version,
        feature_version=args.feature_version,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary.get("errors"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
