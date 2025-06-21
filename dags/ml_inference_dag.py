from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

# Airflow automatically finds plugins in the 'plugins' directory.
from mlflow_sensors import MLflowModelVersionSensor

with DAG(
    dag_id="ml_inference_dag",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    schedule=None,
    tags=["mlops", "inference"],
    doc_md="""
    ### ML Batch Inference Pipeline
    This DAG waits for a version of the 'iris_xgboost_classifier' model
    to be promoted to the 'Staging' stage in the MLflow Model Registry.
    Once the sensor detects the promoted model, it runs the batch inference pipeline.
    """,
) as dag:
    # --- THE FIX: Use our new custom sensor ---
    # This sensor actively polls the MLflow server until the condition is met.
    wait_for_model_in_staging = MLflowModelVersionSensor(
        task_id="wait_for_model_in_staging",
        model_name="iris_xgboost_classifier",  # The model we are waiting for
        target_stage="Staging",  # The stage we are waiting for it to reach
        poke_interval=30,  # Check every 30 seconds
        timeout=3600,  # Timeout after 1 hour
    )

    # Task to run the batch inference pipeline
    run_inference_pipeline = BashOperator(
        task_id="run_inference_pipeline",
        bash_command="python -m pipelines.iris_inference_pipeline",
    )

    # The dependency is now on our custom sensor
    wait_for_model_in_staging >> run_inference_pipeline
