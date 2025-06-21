import mlflow
from pyspark.sql.functions import col
from pyspark.sql import SparkSession

from common.spark_utils import get_spark_session
from common.mlflow_utils import setup_mlflow

# --- Constants ---
DELTA_TABLE_PATH = "s3a://delta/iris"
PREDICTIONS_PATH = "s3a://delta/iris_predictions"
MODEL_NAME = "iris_xgboost_classifier"
MODEL_STAGE = "Staging"  # Or "Production" after promotion in MLflow UI


def run_iris_inference_pipeline(input_data_path=DELTA_TABLE_PATH, output_predictions_path=PREDICTIONS_PATH, model_name=MODEL_NAME, model_stage=MODEL_STAGE, spark: SparkSession = None):
    """
    1. Loads a registered model from the MLflow Model Registry.
    2. Loads a batch of data from a Delta table.
    3. Applies the model for inference using a Pandas UDF.
    4. Saves the predictions back to a new Delta table.
    """
    print("Starting batch inference pipeline...")

    setup_mlflow(experiment_name="stock_market_comparison")
    spark_was_provided = spark is not None
    if not spark_was_provided:
        spark = get_spark_session("IrisInferencePipeline")

    # 1. Load model as a PySpark UDF
    model_uri = f"models:/{model_name}/{model_stage}"
    print(f"Loading model from: {model_uri}")
    # The UDF will now correctly resolve the URI via the tracking server
    predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="double")

    # 2. Load batch data
    print(f"Loading batch data from: {input_data_path}")
    df_batch = spark.read.format("delta").load(input_data_path)

    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # 3. Apply the model
    print("Applying model for batch predictions...")
    df_predictions = df_batch.withColumn(
        "prediction", predict_udf(*[col(c) for c in feature_cols])
    )

    print("Predictions generated:")
    df_predictions.select("target", "prediction").show(10)

    # 4. Save predictions
    print(f"Saving predictions to: {output_predictions_path}")
    (df_predictions.write.format("delta").mode("overwrite").save(output_predictions_path))

    print("Inference pipeline finished successfully.")
    if not spark_was_provided:
        spark.stop()


if __name__ == "__main__":
    run_iris_inference_pipeline()
