from pyspark.sql import SparkSession


def get_spark_session(app_name="MLOps_Platform_MVP"):
    """
    Initializes and returns a SparkSession with Delta Lake and S3 support.
    Configured for the local docker-compose setup.
    """
    # Define the required packages
    delta_package = "io.delta:delta-core_2.12:2.4.0"
    aws_sdk_package = "org.apache.hadoop:hadoop-aws:3.3.4" # For S3A filesystem

    return (
        SparkSession.builder.appName(app_name)
        .config("spark.jars.packages", f"{delta_package},{aws_sdk_package}")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        # Improve performance by caching the packages
        .config("spark.jars.ivy", "/tmp/.ivy2")
        .getOrCreate()
    )
