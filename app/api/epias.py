from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import settings
from app.schemas.epias import (
    EpiasHealthResponse,
    EpiasTestPostRequest,
    EpiasTestPostResponse,
)
from data_pipeline.epias.client import (
    EpiasClient,
    EpiasClientError,
    EpiasCredentialsError,
)
from data_pipeline.epias.repository import (
    RawResponsePersistenceError,
    save_raw_epias_response,
)

router = APIRouter(prefix="/api/epias", tags=["epias"])


@lru_cache
def get_epias_client() -> EpiasClient:
    return EpiasClient()


@router.get("/health", response_model=EpiasHealthResponse)
def epias_health(
    client: EpiasClient = Depends(get_epias_client),
) -> EpiasHealthResponse:
    return EpiasHealthResponse(
        epias_base_url=client.base_url,
        credentials_configured=client.credentials_configured,
        client_ready=True,
    )


@router.post("/test-post", response_model=EpiasTestPostResponse)
def epias_test_post(
    request: EpiasTestPostRequest,
    client: EpiasClient = Depends(get_epias_client),
) -> EpiasTestPostResponse:
    # This generic endpoint is intentionally restricted to local development.
    if settings.environment.lower() not in {"development", "local", "test"}:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="EPİAŞ test endpoint is disabled in this environment.",
        )

    try:
        response = client.post(
            endpoint=request.endpoint,
            payload=request.payload,
            use_auth=request.use_auth,
        )
        response_id = save_raw_epias_response(
            endpoint_name=_endpoint_name(request.endpoint),
            endpoint_url=response.endpoint_url,
            request_payload=request.payload,
            response_json=response.data,
            status_code=response.status_code,
            data_start_date=_payload_date(
                request.payload,
                "startDate",
                "start_date",
            ),
            data_end_date=_payload_date(
                request.payload,
                "endDate",
                "end_date",
            ),
        )
    except EpiasCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except EpiasClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except RawResponsePersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return EpiasTestPostResponse(
        endpoint=request.endpoint,
        status_code=response.status_code,
        raw_response_id=response_id,
        result=response.data,
    )


def _endpoint_name(endpoint: str) -> str:
    return endpoint.rstrip("/").rsplit("/", maxsplit=1)[-1] or "root"


def _payload_date(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value[:10]
    return None
