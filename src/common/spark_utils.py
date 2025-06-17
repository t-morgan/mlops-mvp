import os
from pyspark.sql import SparkSession


def get_spark_session(app_name="MLOps_Platform_MVP"):
    """
    Initializes and returns a SparkSession with Delta Lake and S3 support.
    Configured to run efficiently locally AND be configurable for Airflow.
    """
    delta_package = "io.delta:delta-spark_2.12:3.2.0"
    aws_sdk_package = "org.apache.hadoop:hadoop-aws:3.3.4"

    s3_endpoint_url = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    print(f"Spark configured to use S3 endpoint: {s3_endpoint_url}")

    return (
        SparkSession.builder.appName(app_name)
        # --- Add resource limits ---
        .master("local[2]")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        # ---
        .config("spark.jars.packages", f"{delta_package},{aws_sdk_package}")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
        # --- Use the configurable S3 endpoint ---
        .config("spark.hadoop.fs.s3a.endpoint", s3_endpoint_url)
        # ---
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.jars.ivy", "/tmp/.ivy2")
        .getOrCreate()
    )
