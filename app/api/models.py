from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError

from app.schemas.models import (
    BaselineEvaluationRequest,
    BaselineEvaluationSummary,
    BaselineStatusResponse,
    XGBoostStatusResponse,
    XGBoostTrainingRequest,
    XGBoostTrainingSummary,
)
from ml.models.baseline_ptf import BaselinePtfService
from ml.models.xgboost_ptf import XGBoostPtfService

router = APIRouter(prefix="/api/models", tags=["models"])


def get_baseline_ptf_service() -> BaselinePtfService:
    return BaselinePtfService()


def get_xgboost_ptf_service() -> XGBoostPtfService:
    return XGBoostPtfService()


@router.post("/baseline/ptf/run", response_model=BaselineEvaluationSummary)
def run_baseline_ptf_evaluation(
    request: BaselineEvaluationRequest,
    service: BaselinePtfService = Depends(get_baseline_ptf_service),
) -> BaselineEvaluationSummary:
    try:
        summary = service.run_baseline_evaluation(
            start_date=request.start_date,
            end_date=request.end_date,
        )
        return BaselineEvaluationSummary.model_validate(summary)
    except (SQLAlchemyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.get("/baseline/ptf/status", response_model=BaselineStatusResponse)
def baseline_ptf_status(
    service: BaselinePtfService = Depends(get_baseline_ptf_service),
) -> BaselineStatusResponse:
    try:
        return BaselineStatusResponse.model_validate(service.get_status())
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not query baseline evaluation status.",
        ) from exc


@router.post("/xgboost/ptf/train", response_model=XGBoostTrainingSummary)
def train_xgboost_ptf(
    request: XGBoostTrainingRequest,
    service: XGBoostPtfService = Depends(get_xgboost_ptf_service),
) -> XGBoostTrainingSummary:
    try:
        summary = service.run_training(
            train_start=request.train_start,
            train_end=request.train_end,
            test_start=request.test_start,
            test_end=request.test_end,
            model_version=request.model_version,
            feature_version=request.feature_version,
        )
        return XGBoostTrainingSummary.model_validate(summary)
    except (RuntimeError, SQLAlchemyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.get("/xgboost/ptf/status", response_model=XGBoostStatusResponse)
def xgboost_ptf_status(
    service: XGBoostPtfService = Depends(get_xgboost_ptf_service),
) -> XGBoostStatusResponse:
    try:
        return XGBoostStatusResponse.model_validate(service.get_status())
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not query XGBoost training status.",
        ) from exc
