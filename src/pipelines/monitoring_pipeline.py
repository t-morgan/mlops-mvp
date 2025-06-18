import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from scipy.stats import ks_2samp

from common.spark_utils import get_spark_session

# --- Configuration ---
DELTA_TABLE_PATH = "s3a://delta/iris"
# The Pushgateway service name from docker-compose
PUSHGATEWAY_URL = "pushgateway:9091"
# A unique identifier for our batch job
JOB_NAME = "model_monitoring_batch"


def calculate_data_drift(
    reference_data: pd.DataFrame, production_data: pd.DataFrame
) -> dict:
    """
    Simulates data drift calculation using the Kolmogorov-Smirnov test
    on the 'sepal_length' feature as an example.
    """
    drift_scores = {}
    feature = "sepal_length"

    # The K-S test returns a statistic and a p-value.
    # A small p-value (e.g., < 0.05) suggests significant drift.
    ks_statistic, p_value = ks_2samp(reference_data[feature], production_data[feature])

    drift_scores[f"{feature}_ks_statistic"] = ks_statistic
    drift_scores[f"{feature}_p_value"] = p_value

    print(
        f"Drift check for '{feature}': KS-statistic={ks_statistic:.4f}, p-value={p_value:.4f}"
    )
    return drift_scores


def run_monitoring_pipeline():
    """
    1. Loads reference and "production" data.
    2. Calculates data drift metrics.
    3. Pushes these metrics to the Prometheus Pushgateway.
    """
    print("Starting model monitoring pipeline...")
    spark = get_spark_session("MonitoringPipeline")

    # 1. Load data
    df_spark = spark.read.format("delta").load(DELTA_TABLE_PATH)
    full_df = df_spark.toPandas()

    # Simulate splitting data into reference and production sets
    # In a real scenario, reference_df would be your training data distribution,
    # and production_df would be data logged from live requests.
    reference_df = full_df.sample(n=50, random_state=42)
    production_df = full_df.sample(
        n=50, random_state=1
    )  # Different sample to simulate drift

    # 2. Calculate data drift
    drift_metrics = calculate_data_drift(reference_df, production_df)

    # 3. Push metrics to Pushgateway
    print(f"Pushing metrics to Prometheus Pushgateway at {PUSHGATEWAY_URL}")
    registry = CollectorRegistry()

    # Create Gauge metrics for each drift score
    for key, value in drift_metrics.items():
        # Sanitize the key to be a valid Prometheus metric name
        metric_name = key.replace("-", "_")
        g = Gauge(metric_name, f"Drift metric: {key}", registry=registry)
        g.set(value)

    push_to_gateway(PUSHGATEWAY_URL, job=JOB_NAME, registry=registry)
    print("Metrics pushed successfully.")

    spark.stop()


if __name__ == "__main__":
    run_monitoring_pipeline()
