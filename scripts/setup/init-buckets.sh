#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Use mc's --quiet flag to suppress verbose output in the loop
# The 'until' command will run the 'mc alias' command until it succeeds (returns exit code 0)
until /usr/bin/mc --quiet alias set myminio http://minio:9000 minioadmin minioadmin; do
    echo "Waiting for MinIO to be ready..."
    sleep 1
done

echo "MinIO is ready. Setting up buckets..."

/usr/bin/mc --quiet mb myminio/mlflow --ignore-existing
/usr/bin/mc --quiet mb myminio/delta --ignore-existing

echo "Setting bucket policies..."
/usr/bin/mc --quiet anonymous set public myminio/mlflow
/usr/bin/mc --quiet anonymous set public myminio/delta

echo "MinIO setup complete."