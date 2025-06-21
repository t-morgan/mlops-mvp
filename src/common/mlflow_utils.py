import mlflow
import os
from mlflow.tracking import MlflowClient


def setup_mlflow(experiment_name):
    """
    Sets up the MLflow tracking server and experiment.
    Reads the tracking URI from an environment variable for containerized execution.
    This version is robust against parallel execution race conditions.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

    client = MlflowClient(tracking_uri=tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    experiment = client.get_experiment_by_name(name=experiment_name)
    if experiment is None:
        try:
            print(f"Experiment '{experiment_name}' not found. Creating a new one.")
            client.create_experiment(name=experiment_name)
        except mlflow.exceptions.MlflowException as e:
            if "RESOURCE_ALREADY_EXISTS" in str(e):
                print(
                    f"Experiment '{experiment_name}' was created by a parallel task. Proceeding."
                )
            else:
                raise e

    mlflow.set_experiment(experiment_name=experiment_name)
    print(f"MLflow tracking URI set to: {tracking_uri}")
    print(f"MLFLOW experiment set to: {experiment_name}")
