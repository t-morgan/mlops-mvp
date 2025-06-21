from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
import pendulum

with DAG(
    dag_id="stock_experiment_dag",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    schedule=None,
    tags=["mlops", "stock-trading"],
) as dag:
    # Common data preparation step
    run_data_pipeline = BashOperator(
        task_id="run_stock_data_pipeline",
        bash_command="python -m pipelines.stock_data_pipeline",
    )

    # Branch for XGBoost training
    run_xgboost_pipeline = BashOperator(
        task_id="run_stock_xgboost_pipeline",
        bash_command="python -m pipelines.stock_xgboost_pipeline",
    )

    # Branch for DQL training
    run_dql_pipeline = BashOperator(
        task_id="run_stock_dql_pipeline",
        bash_command="python -m pipelines.stock_dql_pipeline",
    )

    # Define dependencies: Data prep runs first, then the two training jobs run in parallel
    run_data_pipeline >> [run_xgboost_pipeline, run_dql_pipeline]
