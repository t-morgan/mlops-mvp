import mlflow
import os

DEFAULT_MLFLOW_TRACKING_URI = "http://localhost:5000"

def setup_mlflow(
    tracking_uri="http://localhost:5000", experiment_name="iris_xgboost_classification"):
    """
    Sets up the MLflow tracking server and experiment.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI set to: {tracking_uri}")
    print(f"MLflow experiment set to: {experiment_name}")
