import json
import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


class RawResponsePersistenceError(RuntimeError):
    """Raised when a raw EPİAŞ response cannot be persisted."""


def save_raw_epias_response(
    endpoint_name: str,
    endpoint_url: str,
    request_payload: dict[str, Any],
    response_json: Any,
    status_code: int,
    data_start_date: str | None = None,
    data_end_date: str | None = None,
    session: Session | None = None,
) -> int:
    statement = text(
        """
        INSERT INTO raw_epias_responses (
            endpoint_name,
            endpoint_url,
            request_payload,
            response_json,
            status_code,
            data_start_date,
            data_end_date
        )
        VALUES (
            :endpoint_name,
            :endpoint_url,
            CAST(:request_payload AS JSONB),
            CAST(:response_json AS JSONB),
            :status_code,
            CAST(:data_start_date AS DATE),
            CAST(:data_end_date AS DATE)
        )
        RETURNING id
        """
    )
    values = {
        "endpoint_name": endpoint_name,
        "endpoint_url": endpoint_url,
        "request_payload": json.dumps(request_payload),
        "response_json": json.dumps(response_json),
        "status_code": status_code,
        "data_start_date": data_start_date,
        "data_end_date": data_end_date,
    }

    owns_session = session is None
    database_session = session or SessionLocal()
    try:
        response_id = database_session.execute(statement, values).scalar_one()
        database_session.commit()
        logger.info(
            "Stored raw EPİAŞ response: id=%s endpoint=%s status=%s",
            response_id,
            endpoint_name,
            status_code,
        )
        return int(response_id)
    except SQLAlchemyError as exc:
        database_session.rollback()
        logger.exception(
            "Could not store raw EPİAŞ response for endpoint=%s.",
            endpoint_name,
        )
        raise RawResponsePersistenceError(
            "Could not persist the raw EPİAŞ response."
        ) from exc
    finally:
        if owns_session:
            database_session.close()

