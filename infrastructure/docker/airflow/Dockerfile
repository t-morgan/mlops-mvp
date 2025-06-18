ARG AIRFLOW_VERSION=2.8.4

# --- Build Stage ---
FROM python:3.11-slim AS builder

WORKDIR /wheelhouse

COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/

RUN pip install build && python -m build

# --- Final Airflow Stage ---
FROM apache/airflow:${AIRFLOW_VERSION}-python3.11

# Install Java and other system dependencies for PySpark
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
USER airflow

WORKDIR /opt/airflow

COPY requirements/core.txt ./requirements/
COPY requirements/airflow.txt ./requirements/

RUN pip install --no-cache-dir -r ./requirements/airflow.txt --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.11.txt"

COPY --from=builder /wheelhouse/dist/*.whl .
RUN pip install --no-cache-dir ./*.whl && rm ./*.whl

USER root
RUN chown -R airflow: /opt/airflow
USER airflow