from fastapi import FastAPI

from app.api.router import router
from app.core.config import settings


def create_app() -> FastAPI:
    application = FastAPI(
        title=settings.project_name,
        version=settings.app_version,
        description="API for the EPİAŞ PTF Forecasting MVP.",
    )
    application.include_router(router)
    return application


app = create_app()

