from sklearn.datasets import load_iris
from pyspark.sql.types import (
    StructType,
    StructField,
    DoubleType,
    IntegerType,
    StringType,
)
from pyspark.sql import SparkSession

from common.spark_utils import get_spark_session

DELTA_TABLE_PATH = "s3a://delta/iris"


def run_iris_data_pipeline(output_path=DELTA_TABLE_PATH, spark: SparkSession = None):
    print("Starting data pipeline...")
    spark_was_provided = spark is not None
    if not spark_was_provided:
        spark = get_spark_session("IrisDataPipeline")

    # 1. Load raw data from scikit-learn
    print("Loading raw NumPy data from scikit-learn...")
    iris = load_iris()
    # iris.data is a NumPy array of features
    # iris.target is a NumPy array of integer labels
    # iris.target_names is an array of string labels

    # 2. Convert all raw NumPy data into a list of pure Python tuples.
    # This is the most robust way to prepare data for Spark.
    print("Converting raw data into a list of pure Python tuples...")
    data_as_list_of_tuples = []
    for features, target_int in zip(iris.data, iris.target):
        # Explicitly cast every single value to a native Python type
        row_tuple = (
            float(features[0]),  # sepal_length
            float(features[1]),  # sepal_width
            float(features[2]),  # petal_length
            float(features[3]),  # petal_width
            int(target_int),  # target
            str(iris.target_names[target_int]),  # target_name
        )
        data_as_list_of_tuples.append(row_tuple)

    print(
        f"Successfully created a list of {len(data_as_list_of_tuples)} Python tuples."
    )

    # 3. Define the explicit schema. This is still a critical best practice.
    schema = StructType(
        [
            StructField("sepal_length", DoubleType(), True),
            StructField("sepal_width", DoubleType(), True),
            StructField("petal_length", DoubleType(), True),
            StructField("petal_width", DoubleType(), True),
            StructField("target", IntegerType(), True),
            StructField("target_name", StringType(), True),
        ]
    )

    # 4. Create Spark DataFrame from the sanitized list of tuples.
    print("Creating Spark DataFrame from sanitized list of tuples...")
    df_spark = spark.createDataFrame(data_as_list_of_tuples, schema=schema)

    # 5. Perform a robust, JVM-only action first to confirm success.
    # .count() does not require sending data back to Python.
    count = df_spark.count()
    print(f"Spark DataFrame created successfully. Row count: {count}")

    # 6. Now, cautiously try to .show(). If an error happens here, it's on the return trip.
    df_spark.show(5)

    # 7. Write to Delta Lake
    print(f"Writing data to Delta table at: {output_path}")
    (
        df_spark.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(output_path)
    )

    print("Data pipeline finished successfully.")
    if not spark_was_provided:
        spark.stop()


if __name__ == "__main__":
    run_iris_data_pipeline()
