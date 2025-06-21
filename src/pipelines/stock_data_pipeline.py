import pandas as pd
from common.spark_utils import get_spark_session
from sklearn.preprocessing import StandardScaler
from pyspark.sql import SparkSession

RAW_DATA_PATH = "/opt/airflow/data/raw/istanbul_stock_exchange.csv"
DELTA_TABLE_PATH = "s3a://delta/istanbul_stock"


def run_stock_data_pipeline(input_data_path=RAW_DATA_PATH, output_data_path=DELTA_TABLE_PATH, spark: SparkSession = None):
    print(f"Starting Istanbul Stock data pipeline from: {RAW_DATA_PATH}")
    spark_was_provided = spark is not None
    if not spark_was_provided:
        spark = get_spark_session("StockDataPipeline")

    df = pd.read_csv(input_data_path, index_col="date", parse_dates=True)
    df = df.select_dtypes(include="number")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)

    df_scaled["target"] = (df["ISE"].shift(-1) > df["ISE"]).astype(int)

    df_scaled = df_scaled.reset_index()
    df_scaled = df_scaled.dropna()

    print(f"Processed {len(df_scaled)} records.")

    df_spark = spark.createDataFrame(df_scaled)
    (
        df_spark.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(output_data_path)
    )
    print(f"Data saved to Delta table: {output_data_path}")
    if not spark_was_provided:
        spark.stop()


if __name__ == "__main__":
    run_stock_data_pipeline()
