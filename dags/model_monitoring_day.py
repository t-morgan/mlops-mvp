from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="model_monitoring_dag",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    schedule="*/15 * * * *",  # Run every 15 minutes
    tags=["mlops", "monitoring"],
    doc_md="""
    ### Model Monitoring Pipeline
    This DAG periodically runs a job to calculate data drift metrics
    and pushes them to Prometheus for observability.
    """,
) as dag:
    run_monitoring_pipeline = BashOperator(
        task_id="run_monitoring_pipeline",
        bash_command="python -m pipelines.monitoring_pipeline",
    )
