# Use the official MLflow image as a base
FROM ghcr.io/mlflow/mlflow:v2.13.0

# Install the Python package for PostgreSQL connectivity
RUN pip install psycopg2-binary

# This gives the MLflow server a more robust tool for S3 operations.
RUN pip install awscli
