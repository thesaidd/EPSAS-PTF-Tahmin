import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

AUTH_PATH = "/cas/v1/tickets"
TGT_LIFETIME_SECONDS = 2 * 60 * 60
TGT_REFRESH_MARGIN_SECONDS = 5 * 60
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class EpiasClientError(RuntimeError):
    """Base error raised by the EPİAŞ client."""


class EpiasCredentialsError(EpiasClientError):
    """Raised when an authenticated request has no configured credentials."""


class EpiasAuthenticationError(EpiasClientError):
    """Raised when EPİAŞ authentication fails."""


class EpiasRequestError(EpiasClientError):
    """Raised when an EPİAŞ request cannot be completed."""


class EpiasResponseError(EpiasClientError):
    """Raised when EPİAŞ returns an invalid or unsuccessful response."""


@dataclass(frozen=True)
class EpiasResponse:
    endpoint_url: str
    status_code: int
    data: Any


class EpiasClient:
    def __init__(
        self,
        config: Settings | None = None,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or get_settings()
        self.base_url = self.config.epias_base_url.rstrip("/")
        self.auth_url = self.config.epias_auth_url.rstrip("/")
        self.timeout = self.config.epias_request_timeout
        self.max_retries = self.config.epias_max_retries
        self._http_client = http_client or httpx.Client(timeout=self.timeout)
        self._owns_http_client = http_client is None
        self._sleep = sleep
        self._tgt: str | None = None
        self._tgt_expires_at = 0.0
        self._token_lock = threading.Lock()

    @property
    def credentials_configured(self) -> bool:
        return self.config.epias_credentials_configured

    def authenticate(self, force_refresh: bool = False) -> str:
        if not self.credentials_configured:
            raise EpiasCredentialsError(
                "EPİAŞ credentials are not configured. Set EPIAS_USERNAME and "
                "EPIAS_PASSWORD to use authenticated endpoints."
            )

        with self._token_lock:
            if not force_refresh and self._has_valid_token():
                return self._tgt or ""

            auth_endpoint = f"{self.auth_url}{AUTH_PATH}"
            response = self._request_with_retries(
                "POST",
                auth_endpoint,
                headers={
                    "Accept": "text/plain",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "username": self.config.epias_username or "",
                    "password": self.config.epias_password or "",
                },
            )

            if response.status_code != httpx.codes.CREATED:
                raise EpiasAuthenticationError(
                    f"EPİAŞ authentication failed with HTTP {response.status_code}."
                )

            token = response.text.strip()
            if not token:
                raise EpiasAuthenticationError(
                    "EPİAŞ authentication returned an empty TGT token."
                )

            self._tgt = token
            self._tgt_expires_at = (
                time.monotonic()
                + TGT_LIFETIME_SECONDS
                - TGT_REFRESH_MARGIN_SECONDS
            )
            logger.info("EPİAŞ TGT token acquired and cached.")
            return token

    def post(
        self,
        endpoint: str,
        payload: dict[str, Any],
        use_auth: bool = True,
    ) -> EpiasResponse:
        endpoint_url = self._build_endpoint_url(endpoint)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if use_auth:
            headers["TGT"] = self.authenticate()

        response = self._request_with_retries(
            "POST",
            endpoint_url,
            headers=headers,
            json=payload,
        )

        if use_auth and response.status_code in {
            httpx.codes.UNAUTHORIZED,
            httpx.codes.FORBIDDEN,
        }:
            logger.info("EPİAŞ rejected the cached TGT; refreshing it once.")
            headers["TGT"] = self.authenticate(force_refresh=True)
            response = self._request_with_retries(
                "POST",
                endpoint_url,
                headers=headers,
                json=payload,
            )

        if not response.is_success:
            raise EpiasResponseError(
                f"EPİAŞ request to {endpoint_url} failed with "
                f"HTTP {response.status_code}."
            )

        try:
            response_data = response.json()
        except ValueError as exc:
            raise EpiasResponseError(
                f"EPİAŞ returned non-JSON content from {endpoint_url}."
            ) from exc

        return EpiasResponse(
            endpoint_url=endpoint_url,
            status_code=response.status_code,
            data=response_data,
        )

    def close(self) -> None:
        if self._owns_http_client:
            self._http_client.close()

    def _has_valid_token(self) -> bool:
        return bool(self._tgt and time.monotonic() < self._tgt_expires_at)

    def _build_endpoint_url(self, endpoint: str) -> str:
        parsed = urlparse(endpoint)
        if parsed.scheme or parsed.netloc or not endpoint.startswith("/"):
            raise ValueError("EPİAŞ endpoint must be a relative path starting with '/'.")
        return f"{self.base_url}{endpoint}"

    def _request_with_retries(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            try:
                response = self._http_client.request(
                    method,
                    url,
                    timeout=self.timeout,
                    **kwargs,
                )
            except httpx.RequestError as exc:
                if attempt >= self.max_retries:
                    raise EpiasRequestError(
                        f"EPİAŞ request to {url} failed after "
                        f"{attempt + 1} attempt(s)."
                    ) from exc
                self._wait_before_retry(url, attempt, reason=type(exc).__name__)
                continue

            if (
                response.status_code not in RETRYABLE_STATUS_CODES
                or attempt >= self.max_retries
            ):
                logger.info(
                    "EPİAŞ request completed: url=%s status=%s attempt=%s",
                    url,
                    response.status_code,
                    attempt + 1,
                )
                return response

            self._wait_before_retry(
                url,
                attempt,
                reason=f"HTTP {response.status_code}",
            )

        raise EpiasRequestError(f"EPİAŞ request to {url} failed unexpectedly.")

    def _wait_before_retry(self, url: str, attempt: int, reason: str) -> None:
        delay_seconds = min(0.5 * (2**attempt), 8.0)
        logger.warning(
            "Retrying EPİAŞ request: url=%s reason=%s delay_seconds=%s",
            url,
            reason,
            delay_seconds,
        )
        self._sleep(delay_seconds)

