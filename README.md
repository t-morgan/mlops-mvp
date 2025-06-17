# MLOps Monorepo Platform: PySpark + MLflow + Airflow

This repository contains a comprehensive, production-style MLOps platform built as a monorepo. It integrates PySpark for distributed data processing, MLflow for end-to-end machine learning lifecycle management, and Apache Airflow for robust pipeline orchestration.

This MVP demonstrates a complete, orchestrated ML pipeline from data ingestion to batch inference, showcasing a core developer workflow for building, tracking, deploying, and using models in a robust, reproducible, and automated environment.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Monorepo Structure](#monorepo-structure)
- [Local Development Setup](#local-development-setup)
- [Running the Orchestrated Pipeline with Airflow](#running-the-orchestrated-pipeline-with-airflow)
- [Manual Development Workflow (Without Airflow)](#manual-development-workflow-without-airflow)
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
* **Workflow Orchestration**:
* **Apache Airflow**: For scheduling, executing, and monitoring the entire ML pipeline, including steps that require manual intervention.
* **Infrastructure (Local Simulation)**:
* **Docker Compose**: To orchestrate all backend services for local development.
* **MinIO**: As an S3-compatible object store for MLflow artifacts and Delta Lake tables.
* **PostgreSQL**: As a robust backend store for both MLflow and Airflow metadata.

![Architecture Diagram](docs/architecture.png)
*(Note: You can create a simple diagram using a tool like diagrams.net and place it in the `docs/` folder)*

## Monorepo Structure

The repository is structured to separate concerns, making it scalable and easy to maintain.

```
mlops-platform/
├── dags/ # Airflow DAG definitions
├── infrastructure/ # Container definitions (Docker)
├── plugins/ # Custom Airflow plugins (e.g., sensors)
├── scripts/ # Helper scripts for setup, deployment, etc.
├── src/ # Main source code for pipelines and utilities
└── ... (other folders like docs, tests, notebooks)
```

## Local Development Setup

Follow these steps to set up the complete MLOps environment on your local machine.

### Prerequisites

* [Docker](https://www.docker.com/products/docker-desktop/) and Docker Compose
* [Python 3.11+](https://www.python.org/downloads/)
* `git`

### Step 1: Clone the Repository & Configure Credentials

```bash
git clone <your-repo-url>
cd mlops-platform

# For local development, boto3 (used by MLflow and Spark) needs to know
# how to connect to MinIO. The cleanest way is a config file.
# Create the config directory:
mkdir -p ~/.aws

# Create the config file (~/.aws/config):
cat > ~/.aws/config << EOL
[default]
region = us-east-1
s3 =
endpoint_url = http://localhost:9000
signature_version = s3v4
s3api =
endpoint_url = http://localhost:9000
EOL

# Create the credentials file (~/.aws/credentials):
cat > ~/.aws/credentials << EOL
[default]
aws_access_key_id = minioadmin
aws_secret_access_key = minioadmin
EOL
```

### Step 2: Set Up Python Virtual Environment

It is crucial to use a virtual environment to manage dependencies for local testing and IDE support.

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it (on macOS/Linux)
source venv/bin/activate
# On Windows: venv\Scripts\activate

# Install all required packages, including Airflow for IDE support
pip install -r requirements.txt

# Install the local 'src' code as an editable package
pip install -e .
```

### Step 3: Start All Backend Services

This command builds the custom Docker images and starts the entire backend stack: MLflow, MinIO, PostgreSQL, and Airflow.

```bash
docker-compose up -d --build
```
*Note: The first build may take several minutes as it installs Java and Python dependencies.*

After startup, you can access the services:
* **Airflow UI**: `http://localhost:8080` (Login: `admin` / `admin`)
* **MLflow UI**: `http://localhost:5000`
* **MinIO Console**: `http://localhost:9001` (Login: `minioadmin` / `minioadmin`)

## Running the Orchestrated Pipeline with Airflow

This is the primary, intended workflow for the platform.

### Step 1: Trigger the Training DAG

1. In the Airflow UI (`http://localhost:8080`), find and un-pause the **`ml_training_dag`**.
2. Click the "Play" button (▶️) to trigger a new run.
3. Watch as the `run_data_pipeline` and `run_training_pipeline` tasks execute and succeed. This will create a new model version in MLflow, staged as "None".

### Step 2: Perform Manual Model Review and Promotion

This step simulates the critical human-in-the-loop validation process.

1. Go to the MLflow UI (`http://localhost:5000`) and review the latest run's metrics and artifacts.
2. If the model is acceptable, navigate to the **Models** tab, find the latest version of `iris_xgboost_classifier`, and use the **Stage** dropdown to promote it to **Staging**.

### Step 3: Trigger the Inference DAG

1. In the Airflow UI, find and un-pause the **`ml_inference_dag`**.
2. Trigger a new run.
3. The first task, `wait_for_model_in_staging`, will start. It will poll the MLflow server every 30 seconds.
4. Once you complete Step 2, this sensor will detect the change, turn green, and automatically trigger the final `run_inference_pipeline` task.

You have now successfully executed the entire orchestrated pipeline, including a manual validation gate.

## Manual Development Workflow (Without Airflow)

For rapid development and debugging of a single script, you can run them directly from your local machine.

1. Ensure services are running: `docker-compose up -d`.
2. Activate your `venv`: `source venv/bin/activate`.
3. Run any pipeline script directly:
```bash
# Example: Run just the training pipeline
python src/pipelines/training_pipeline.py
```
The code is designed to use `localhost` endpoints when run locally and container service names when run via Airflow.

## Core Concepts Demonstrated

- **Infrastructure as Code**: All services are defined and configured in `docker-compose.yml`.
- **Immutable Infrastructure**: Custom Docker images bake in all system and Python dependencies for a reproducible runtime environment.
- **Orchestration**: Airflow manages the entire workflow, including dependencies and manual gates.
- **Custom Extensibility**: A custom Airflow Sensor (`plugins/mlflow_sensors.py`) integrates directly with an external system (MLflow) to control pipeline flow.
- **Configuration as Code**: Key configurations (like URIs) are managed with environment variables, separating them from the application logic.
- **Separation of Concerns**: The two-DAG approach cleanly separates the model training lifecycle from the model deployment/inference lifecycle.
