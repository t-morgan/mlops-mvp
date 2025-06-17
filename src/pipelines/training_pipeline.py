import mlflow
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from common.spark_utils import get_spark_session
from common.mlflow_utils import setup_mlflow

# Path to the data created by the data pipeline
DELTA_TABLE_PATH = "s3a://delta/iris"
MODEL_NAME = "iris_xgboost_classifier"


def run_training_pipeline():
    """
    1. Reads data from the Delta table.
    2. Trains an XGBoost classifier.
    3. Logs parameters, metrics, and the model to MLflow.
    4. Registers the model in the MLflow Model Registry.
    """
    print("Starting training pipeline...")
    setup_mlflow()
    spark = get_spark_session("TrainingPipeline")

    # 1. Read data from Delta Lake
    print(f"Reading data from Delta table: {DELTA_TABLE_PATH}")
    df_spark = spark.read.format("delta").load(DELTA_TABLE_PATH)
    df_pandas = df_spark.toPandas()
    print(f"Successfully loaded {len(df_pandas)} records.")

    # 2. Feature Engineering & Splitting
    X = df_pandas[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df_pandas["target"]

    # XGBoost requires numeric labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # MLflow Tracking
    with mlflow.start_run(run_name="XGBoost_Training_Run") as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_param("data_path", DELTA_TABLE_PATH)

        # 3. Train Model
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": 4,
            "eta": 0.1,
            "eval_metric": "mlogloss",
        }
        mlflow.log_params(params)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtest, "test")],
            early_stopping_rounds=10,
            verbose_eval=False,
        )

        # 4. Evaluate and Log Metrics
        y_pred_proba = model.predict(dtest)
        y_pred = y_pred_proba.argmax(axis=1)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Model Accuracy: {accuracy:.4f}")
        mlflow.log_metric("accuracy", accuracy)

        # 5. Log Model and Register it
        print("Logging and registering the model...")
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,  # This will create and register the model
        )
        print(f"Model '{MODEL_NAME}' registered in MLflow Model Registry.")

    print("Training pipeline finished successfully.")
    spark.stop()


if __name__ == "__main__":
    run_training_pipeline()
