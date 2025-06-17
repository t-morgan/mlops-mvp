# MLOps Monorepo Platform: PySpark + MLflow

This repository contains a comprehensive, production-style MLOps platform built as a monorepo. It integrates PySpark for distributed data processing with MLflow for end-to-end machine learning lifecycle management.

This MVP demonstrates a complete ML pipeline from data ingestion to batch inference, showcasing a core developer workflow for building, tracking, and using models in a robust, reproducible, and scalable environment.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Monorepo Structure](#monorepo-structure)
- [Local Development Setup](#local-development-setup)
- [Running the MVP Pipeline](#running-the-mvp-pipeline)
- [Core Concepts Demonstrated](#core-concepts-demonstrated)
- [Next Steps & Production Considerations](#next-steps--production-considerations)

## Architecture Overview

This platform is designed around a set of core, open-source tools orchestrated for a seamless MLOps experience.

* **Data Layer**:
* **Apache Spark (PySpark)**: For scalable, distributed data processing.
* **Delta Lake**: For creating versioned, ACID-compliant data lakes (simulated on MinIO).
* **ML Lifecycle Management**:
* **MLflow Tracking**: For logging experiments, parameters, metrics, and code versions.
* **MLflow Model Registry**: For versioning, staging, and managing trained models.
* **MLflow Projects**: For packaging code for reproducible runs.
* **Infrastructure (Local Simulation)**:
* **Docker Compose**: To orchestrate all backend services for local development.
* **MinIO**: As an S3-compatible object store for MLflow artifacts and Delta Lake tables.
* **PostgreSQL**: As a robust backend store for MLflow metadata (experiments, models).

![Architecture Diagram](docs/architecture.png)
*(Note: You can create a simple diagram using a tool like diagrams.net and place it in the `docs/` folder)*

## Monorepo Structure

The repository is structured to separate concerns, making it scalable and easy to maintain.

```
mlops-platform/
├── .github/ # CI/CD workflows (e.g., testing, deployment)
├── configs/ # Environment/tool specific configurations
├── docs/ # Project documentation
├── infrastructure/ # IaC (Terraform) and container definitions (Docker)
├── notebooks/ # Jupyter notebooks for exploration and analysis
├── scripts/ # Helper scripts for setup, deployment, etc.
├── src/ # Main source code for pipelines and utilities
└── tests/ # Unit, integration, and end-to-end tests
```

For a detailed file-by-file breakdown, see the `docs/` directory.

## Local Development Setup

Follow these steps to set up the complete MLOps environment on your local machine.

### Prerequisites

* [Docker](https://www.docker.com/products/docker-desktop/) and Docker Compose
* [Python 3.9+](https://www.python.org/downloads/)
* `git`

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd mlops-platform
```

### Step 2: Start Backend Services

This command uses Docker Compose to build and start the MLflow server, PostgreSQL database, and MinIO object store.

```bash
docker-compose up -d --build
```

After startup, you can access the services:
* **MLflow UI**: `http://localhost:5000`
* **MinIO Console**: `http://localhost:9001` (Login with `minioadmin` / `minioadmin`)

### Step 3: Set Up Python Virtual Environment

It is crucial to use a virtual environment to manage dependencies and avoid conflicts.

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it (on macOS/Linux)
source venv/bin/activate
# On Windows: venv\Scripts\activate

# Install all required packages
pip install -r requirements.txt

# Install the local 'src' code as an editable package
pip install -e .
```

## Running the MVP Pipeline

This demonstrates the end-to-end workflow. Each script should be run from the root of the project directory.

### 1. Data Processing Pipeline

This script uses PySpark to ingest the Iris dataset, process it, and save it as a Delta table in MinIO.

```bash
python src/pipelines/data_pipeline.py
```
* **Verify**: Check the MinIO console (`http://localhost:9001`). You should see a `delta` bucket containing the `iris/` table.

### 2. Model Training Pipeline

This script reads the data from the Delta table, trains an XGBoost model, and logs the experiment, artifacts, and registered model to MLflow.

**Note**: This script requires credentials to upload model artifacts directly to MinIO.

```bash
AWS_ACCESS_KEY_ID=minioadmin \
AWS_SECRET_ACCESS_KEY=minioadmin \
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
python src/pipelines/training_pipeline.py
```
* **Verify**: Check the MLflow UI (`http://localhost:5000`). You will see a new run and a new registered model named `iris_xgboost_classifier`.

### 3. Promote the Model

In a real workflow, a model would be validated before promotion. For this MVP, we will promote it manually.
1. Go to the **Models** tab in the MLflow UI.
2. Click on `iris_xgboost_classifier`.
3. Click on **Version 1**.
4. Use the **Stage** dropdown to transition the model to **Staging**.

### 4. Batch Inference Pipeline

This script loads the `Staging` model from the registry and uses it to perform batch inference on the Iris dataset.

```bash
AWS_ACCESS_KEY_ID=minioadmin \
AWS_SECRET_ACCESS_KEY=minioadmin \
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
python src/pipelines/inference_pipeline.py
```
* **Verify**: Check the MinIO console. You will now see a new table at `delta/iris_predictions/` containing the original data plus a `prediction` column.

## Core Concepts Demonstrated

- **Infrastructure as Code**: All services are defined in `docker-compose.yml`.
- **Data Versioning**: Using Delta Lake to store and manage datasets.
- **Experiment Tracking**: Using MLflow to log every aspect of a training run.
- **Model Registry**: A central repository for managing and versioning model lifecycle.
- **Reproducibility**: Packaging code and dependencies ensures that pipelines can be re-run reliably.
- **Separation of Concerns**: The monorepo cleanly separates infrastructure, application code, tests, and documentation.

## Next Steps & Production Considerations

This MVP provides a solid foundation. A full production system would expand on this with:
* **CI/CD Automation**: Implement GitHub Actions (`.github/workflows`) to run tests on PRs and automate training/deployment pipelines.
* **Cloud Deployment**: Use Terraform (`infrastructure/terraform`) to provision managed services (e.g., EKS/GKE, RDS, S3) in the cloud.
* **Workflow Orchestration**: Use a tool like Apache Airflow to schedule and manage dependencies between the data, training, and inference pipelines.
* **Real-time Serving**: Deploy the model as a REST API using MLflow's serving capabilities or a dedicated framework like FastAPI, running as a service in Kubernetes.
* **Monitoring**: Implement pipelines to monitor for data drift, concept drift, and model performance degradation.
* **Comprehensive Testing**: Add unit, integration, and end-to-end tests in the `tests/` directory.