# MLOps Monorepo Platform: PySpark + MLflow + Airflow

This repository contains a comprehensive, production-style MLOps platform built as a monorepo. It integrates PySpark for distributed data processing, MLflow for end-to-end machine learning lifecycle management, and Apache Airflow for robust pipeline orchestration.

This MVP demonstrates a complete, orchestrated ML pipeline from data ingestion to batch inference, showcasing a core developer workflow for building, tracking, deploying, and using models in a robust, reproducible, and automated environment.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Monorepo Structure](#monorepo-structure)
- [Local Development Setup](#local-development-setup)
- [Running the Full MLOps Lifecycle](#running-the-full-mlops-lifecycle)
- [Observability & Monitoring](#observability--monitoring)
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
* **Real-Time Serving**:
* **FastAPI**: A high-performance web framework for building APIs.
* **Uvicorn**: An ASGI server to run the FastAPI application.

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

Our services (Airflow and FastAPI) have conflicting dependencies. It is impossible to create a single local environment for all of them. The recommended approach is to set up your local `venv` to match the **FastAPI server** for rapid API development. Airflow will be managed entirely by Docker.

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies for the API server and our core ML logic
pip install -r requirements/api.txt

# Install the local 'src' code as an editable package
pip install -e .
```
Your IDE will now provide full support for editing the API and core pipeline code. It may show an "unresolved import" error for `apache-airflow` in your DAG files; this is expected and can be ignored.

### Step 3: Start All Backend Services

The first time we set up the environment, we need to initialize Airflow

```bash
docker-compose up airflow-init
```

This command builds the custom Docker images and starts the entire backend stack: MLflow, MinIO, PostgreSQL, and Airflow.

```bash
docker-compose up -d --build
```
*Note: The first build may take several minutes as it installs Java and Python dependencies.*

After startup, you can access the services:
* **Airflow UI**: `http://localhost:8080` (Login: `admin` / `admin`)
* **MLflow UI**: `http://localhost:5000`
* **MinIO Console**: `http://localhost:9001` (Login: `minioadmin` / `minioadmin`)

## Running the Full MLOps Lifecycle

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

### Step 4: Promote Model to Production

The final step in the model lifecycle is to promote our validated "Staging" model to "Production".

1. In the MLflow UI, go to the model version you previously promoted to "Staging".
2. Use the **Stage** dropdown to transition it to **Production**.
3. Restart the FastAPI server to pick up the new model:
```bash
docker-compose restart fastapi-server
```
The server is designed to load the "Production" model on startup.

### Step 5: Test the Real-time API

Our FastAPI server is now serving the production model.

1. **Interactive API Docs (Swagger UI)**:
* Open your browser to `http://localhost:8001/docs`.
* You can explore the endpoints, view the schemas, and even send test requests directly from the UI.

2. **Using `curl` from your Terminal**:
* Send a sample prediction request:
```bash
curl -X POST "http://localhost:8001/predict" \
-H "Content-Type: application/json" \
-d '{
"sepal_length": 5.1,
"sepal_width": 3.5,
"petal_length": 1.4,
"petal_width": 0.2
}'
```
* You should receive a response like:
```json
{"prediction":0}
```

### Step 6: Monitor the System

1. In the Airflow UI, un-pause the **`model_monitoring_dag`**. It will now run every 15 minutes, calculating drift metrics.
2. Go to the Grafana UI (`http://localhost:3000`) to view the health of the API server and the model drift metrics pushed by the monitoring pipeline.

## Observability & Monitoring

The platform includes a full observability stack using Prometheus and Grafana to monitor both service health and model performance.

**Services:**
* **Prometheus**: (`http://localhost:9090`) Scrapes and stores metrics.
* **Grafana**: (`http://localhost:3000`) Visualizes metrics in dashboards (Login: `admin` / `admin`).
* **Pushgateway**: (`http://localhost:9091`) A cache for metrics from short-lived jobs.

### Available Metrics

1. **FastAPI Server Health**: The API server automatically exposes metrics at its `/metrics` endpoint. These include:
* `fastapi_requests_total`: Total number of requests.
* `fastapi_requests_latency_seconds`: Latency histogram of requests.
* `fastapi_requests_in_progress`: Number of currently active requests.

2. **Model Data Drift**: The `model_monitoring_dag` runs every 15 minutes. It calculates a Kolmogorov-Smirnov (K-S) statistic and p-value for the `sepal_length` feature to detect data drift and pushes these metrics to Prometheus.

### Setting Up a Grafana Dashboard

1. Go to the Grafana UI at `http://localhost:3000`.
2. **Add Data Source**:
* Go to Connections -> Data sources -> Add new data source.
* Select **Prometheus**.
* For the "Prometheus server URL", enter `http://prometheus:9090`.
* Click "Save & Test". You should see a "Data source is working" message.
3. **Create a Dashboard**:
* Go to Dashboards -> New Dashboard -> Add visualization.
* Select your "Prometheus" data source.
* In the "Metrics browser" input, you can now type `sepal_length` to see the drift metrics, or `fastapi` to see the API health metrics.
* Example Query for p-value: `sepal_length_p_value`
* Example Query for API request rate: `rate(fastapi_requests_total[5m])`

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

- **End-to-End MLOps Lifecycle**: The project covers the full cycle: data -> training -> validation -> deployment -> monitoring -> retraining.
- **Infrastructure as Code**: All services (ML, orchestration, observability) are defined and configured in `docker-compose.yml`.
- **Immutable Infrastructure**: Custom Docker images bake in all system and Python dependencies for a reproducible runtime environment.
- **Workflow Orchestration**: Airflow manages the entire workflow, including dependencies, scheduling, and gates for manual intervention.
- **Custom Extensibility**: A custom Airflow Sensor (`plugins/mlflow_sensors.py`) integrates directly with an external system (MLflow) to control pipeline flow.
- **Separation of Concerns**: Decoupled DAGs for training and inference, and separate dependency files for each service, demonstrate a robust, scalable architecture.
- **Configuration as Code**: Key configurations (like URIs and endpoints) are managed with environment variables, separating them from application logic.
- **Full-Stack Observability**: The Prometheus/Grafana stack provides monitoring for both low-level service metrics (latency, errors) and high-level ML metrics (data drift).

## Next Steps & Production Considerations

This MVP provides a solid foundation. A full production system would expand on this with:
* **CI/CD Automation**: Implement GitHub Actions (`.github/workflows`) to run tests on PRs and automate the building and deployment of Docker images to a container registry (e.g., ECR, GCR).
* **Cloud Deployment**: Use Terraform (`infrastructure/terraform`) to provision managed services (e.g., EKS/GKE for compute, RDS for databases, S3 for storage) in the cloud.
* **Advanced Monitoring & Alerting**:
* Expand the `monitoring_pipeline` to calculate concept drift and model performance metrics (e.g., accuracy, F1-score) by joining predictions with ground truth labels.
* Set up alerting rules in Prometheus/Grafana to notify the team of API errors, high latency, or significant model drift.
* **Scalable Spark Execution**: For large datasets, configure Airflow to submit Spark jobs to a dedicated Spark cluster on Kubernetes or a cloud service (e.g., EMR, Databricks) instead of running Spark in the worker container.
* **Comprehensive Testing**: Add a full suite of unit, integration, and end-to-end tests in the `tests/` directory.
* **Secret Management**: Replace hardcoded credentials with a secure secret management solution like HashiCorp Vault, AWS Secrets Manager, or Doppler.
* **Blue-Green/Canary Deployments**: Implement more sophisticated API deployment strategies instead of a simple restart to ensure zero downtime.