import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.gpr_residual_ptf import GprResidualPtfService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the GPR residual uncertainty model."
    )
    parser.add_argument("--xgboost-training-run-id", default=None)
    parser.add_argument("--residual-train-start", type=date.fromisoformat, default=None)
    parser.add_argument("--residual-train-end", type=date.fromisoformat, default=None)
    parser.add_argument("--residual-test-start", type=date.fromisoformat, default=None)
    parser.add_argument("--residual-test-end", type=date.fromisoformat, default=None)
    parser.add_argument("--model-version", default="gpr_residual_v1")
    parser.add_argument("--max-train-rows", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary: dict[str, Any] = GprResidualPtfService().run_residual_modeling(
        xgboost_training_run_id=args.xgboost_training_run_id,
        residual_train_start=args.residual_train_start,
        residual_train_end=args.residual_train_end,
        residual_test_start=args.residual_test_start,
        residual_test_end=args.residual_test_end,
        model_version=args.model_version,
        max_train_rows=args.max_train_rows,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary.get("errors"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
