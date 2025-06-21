import pytest
from pyspark.sql import SparkSession
import tempfile
import os

@pytest.fixture(scope="session")
def spark_session():
    """
    Creates a Spark session for testing purposes, now with correct Delta Lake packages.
    This session runs in local mode and is scoped to the entire test session.
    """
    delta_package = "io.delta:delta-spark_2.12:3.2.0"
    
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("MLOps-Test-Session")
        .config("spark.jars.packages", delta_package)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture
def local_mlflow_tracking_uri():
    """
    Creates a temporary directory for MLflow tracking and yields the URI.
    This simulates a local MLflow server without needing a real server.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        uri = f"file://{tmp_dir}"
        # Set environment variable so that our util function picks it up
        os.environ["MLFLOW_TRACKING_URI"] = uri
        yield uri
        # Clean up the environment variable
        del os.environ["MLFLOW_TRACKING_URI"]