import mlflow


def setup_mlflow(
    tracking_uri="http://localhost:5000", experiment_name="iris_xgboost_classification"
):
    """
    Sets up the MLflow tracking server and experiment.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI set to: {tracking_uri}")
    print(f"MLflow experiment set to: {experiment_name}")
