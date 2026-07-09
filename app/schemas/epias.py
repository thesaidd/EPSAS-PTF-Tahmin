from typing import Any

from pydantic import BaseModel, Field, field_validator


class EpiasHealthResponse(BaseModel):
    epias_base_url: str
    credentials_configured: bool
    client_ready: bool


class EpiasTestPostRequest(BaseModel):
    endpoint: str
    payload: dict[str, Any] = Field(default_factory=dict)
    use_auth: bool = True

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        if not value.startswith("/") or "://" in value:
            raise ValueError("endpoint must be a relative path starting with '/'")
        return value


class EpiasTestPostResponse(BaseModel):
    endpoint: str
    status_code: int
    raw_response_id: int
    result: Any

