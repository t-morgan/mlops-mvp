import pytest
from pathlib import Path
from mlflow.tracking import MlflowClient

# Import the refactored pipeline functions
# NOTE: You will need to implement the refactoring in your actual pipeline files.
# For this example, we assume they have been refactored.
from pipelines.iris_data_pipeline import run_iris_data_pipeline
from pipelines.iris_training_pipeline import run_iris_training_pipeline
from pipelines.iris_inference_pipeline import run_iris_inference_pipeline

# A marker to indicate this is a slow-running integration test
@pytest.mark.integration
def test_full_iris_workflow(spark_session, tmp_path, local_mlflow_tracking_uri):
    """
    Tests the complete Iris data -> train -> inference workflow.
    
    Args:
        spark_session: Fixture for the Spark session.
        tmp_path: Pytest fixture for a temporary directory.
        local_mlflow_tracking_uri: Fixture for a local MLflow setup.
    """
    # 1. Define paths and names for this test run
    data_path = str(tmp_path / "delta/iris")
    predictions_path = str(tmp_path / "delta/iris_predictions")
    model_name = "test_iris_classifier"

    # 2. --- Run Data Pipeline ---
    run_iris_data_pipeline(output_path=data_path, spark=spark_session)
    
    # Assert: Check that the Delta table was actually created
    assert Path(data_path).exists()
    assert len(list(Path(data_path).glob("*.parquet"))) > 0

    # 3. --- Run Training Pipeline ---
    run_iris_training_pipeline(data_path=data_path, model_name=model_name, spark=spark_session)
    
    # Assert: Check that the model was registered in MLflow
    client = MlflowClient()
    registered_model = client.get_registered_model(model_name)
    assert registered_model.name == model_name
    latest_version = registered_model.latest_versions[0]
    assert latest_version.version == 1

    # 4. --- Simulate Manual Promotion ---
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version.version,
        stage="Staging"
    )

    # 5. --- Run Inference Pipeline ---
    run_iris_inference_pipeline(
        input_data_path=data_path,
        output_predictions_path=predictions_path,
        model_name=model_name,
        model_stage="Staging",
        spark=spark_session,
    )

    # Assert: Check that the predictions table was created
    assert Path(predictions_path).exists()
    predictions_df = spark_session.read.format("delta").load(predictions_path)
    assert "prediction" in predictions_df.columns
    assert predictions_df.count() > 0