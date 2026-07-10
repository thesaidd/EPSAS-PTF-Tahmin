FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install plotly==5.24.1

COPY . .

FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS dashboard
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]

FROM base AS mlflow
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--artifacts-destination", "/mlartifacts", "--serve-artifacts"]
