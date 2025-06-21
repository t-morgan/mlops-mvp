from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="ml_training_dag",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    schedule="@daily",  # Or on a trigger
    tags=["mlops", "training"],
    doc_md="""
    ### ML Training Pipeline
    This DAG ingests data, trains a model, and registers it in MLflow.
    It is the first part of the end-to-end ML workflow.
    """,
) as dag:
    # Task to run the data processing pipeline
    run_data_pipeline = BashOperator(
        task_id="run_data_pipeline",
        bash_command="python -m pipelines.iris_data_pipeline",
    )

    # Task to run the model training pipeline
    run_training_pipeline = BashOperator(
        task_id="run_training_pipeline",
        bash_command="python -m pipelines.iris_training_pipeline",
    )

    # Define the task dependencies
    run_data_pipeline >> run_training_pipeline
