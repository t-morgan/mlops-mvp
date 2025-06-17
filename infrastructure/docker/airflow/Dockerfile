ARG AIRFLOW_VERSION=2.8.4

# --- Build Stage ---
# This stage prepares our installable Python package (a .whl file)
FROM python:3.11-slim AS builder

WORKDIR /wheelhouse

# Copy files needed to build the wheel
COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/

# Install build dependencies and create the wheel file
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

# Install Python dependencies from requirements.txt first
COPY requirements.txt .
RUN pip install --no-cache-dir -r ./requirements.txt --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.11.txt"

# Copy the built wheel from the 'builder' stage
COPY --from=builder /wheelhouse/dist/*.whl .

# Copy plugins
COPY plugins/ ./plugins/

# Install our project from the wheel file
RUN pip install --no-cache-dir ./*.whl && \
    # Clean up the wheel file after installation
    rm ./*.whl

# Set the ownership of the home directory to the airflow user.
USER root
RUN chown -R airflow: /opt/airflow
USER airflow