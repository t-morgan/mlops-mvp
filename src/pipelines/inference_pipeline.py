import mlflow
import pandas as pd
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import DoubleType

from common.spark_utils import get_spark_session
from common.mlflow_utils import setup_mlflow

# --- Constants ---
DELTA_TABLE_PATH = "s3a://delta/iris"
PREDICTIONS_PATH = "s3a://delta/iris_predictions"
MODEL_NAME = "iris_xgboost_classifier"
MODEL_STAGE = "Staging"  # Or "Production" after promotion in MLflow UI


def run_inference_pipeline():
    """
    1. Loads a registered model from the MLflow Model Registry.
    2. Loads a batch of data from a Delta table.
    3. Applies the model for inference using a Pandas UDF.
    4. Saves the predictions back to a new Delta table.
    """
    print("Starting batch inference pipeline...")

    setup_mlflow()

    spark = get_spark_session("InferencePipeline")

    # 1. Load model as a PySpark UDF
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    print(f"Loading model from: {model_uri}")
    # The UDF will now correctly resolve the URI via the tracking server
    predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="double")

    # 2. Load batch data
    print(f"Loading batch data from: {DELTA_TABLE_PATH}")
    df_batch = spark.read.format("delta").load(DELTA_TABLE_PATH)

    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # 3. Apply the model
    print("Applying model for batch predictions...")
    df_predictions = df_batch.withColumn(
        "prediction", predict_udf(*[col(c) for c in feature_cols])
    )

    print("Predictions generated:")
    df_predictions.select("target", "prediction").show(10)

    # 4. Save predictions
    print(f"Saving predictions to: {PREDICTIONS_PATH}")
    (df_predictions.write.format("delta").mode("overwrite").save(PREDICTIONS_PATH))

    print("Inference pipeline finished successfully.")
    spark.stop()


if __name__ == "__main__":
    run_inference_pipeline()
