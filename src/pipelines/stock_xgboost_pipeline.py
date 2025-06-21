import mlflow
import xgboost as xgb
from sklearn.metrics import accuracy_score
from common.spark_utils import get_spark_session
from common.mlflow_utils import setup_mlflow

DELTA_TABLE_PATH = "s3a://delta/istanbul_stock"
MODEL_NAME = "stock_xgboost_predictor"
MINIMUM_ACCURACY_THRESHOLD = 0.52  # A simple performance gate


def run_stock_xgboost_pipeline(data_path=DELTA_TABLE_PATH, model_name=MODEL_NAME):
    """
    A robust training pipeline that:
    1. Splits data into train, validation, and test sets.
    2. Uses the validation set for early stopping to prevent overfitting.
    3. Uses the test set for a final, unbiased performance evaluation.
    4. Implements a performance gate to only register models that meet a threshold.
    """
    setup_mlflow(experiment_name="stock_market_comparison")
    spark = get_spark_session("StockXGBoost")

    df_pandas = (
        spark.read.format("delta")
        .load(data_path)
        .toPandas()
        .set_index("date")
        .sort_index()
    )

    # 1. Split data chronologically: 70% train, 15% validation, 15% test
    train_size = int(len(df_pandas) * 0.70)
    val_size = int(len(df_pandas) * 0.15)

    train_df = df_pandas.iloc[:train_size]
    val_df = df_pandas.iloc[train_size : train_size + val_size]
    test_df = df_pandas.iloc[train_size + val_size :]

    print(
        f"Data split into: Train ({len(train_df)}), Validation ({len(val_df)}), Test ({len(test_df)})"
    )

    # Prepare feature and target sets
    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]
    X_val = val_df.drop("target", axis=1)
    y_val = val_df["target"]
    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]

    with mlflow.start_run(run_name="XGBoost_With_Validation") as run:
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("data_split", "70-15-15")
        mlflow.log_param("min_accuracy_for_registration", MINIMUM_ACCURACY_THRESHOLD)

        # 2. Use validation set for early stopping
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            early_stopping_rounds=10,  # Enable early stopping
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],  # Provide the validation set
            verbose=False,
        )

        # 3. Use the UNSEEN test set for final evaluation
        print("Evaluating model on the hold-out test set...")
        test_preds = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_preds)

        print(f"Final Test Set Accuracy: {test_accuracy:.4f}")
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("best_iteration", model.best_iteration)

        # 4. Implement the performance gate
        if test_accuracy >= MINIMUM_ACCURACY_THRESHOLD:
            print(
                f"Test accuracy ({test_accuracy:.4f}) is above threshold ({MINIMUM_ACCURACY_THRESHOLD}). Registering model..."
            )
            mlflow.xgboost.log_model(
                xgb_model=model, artifact_path="model", registered_model_name=model_name
            )
            print(f"Model '{model_name}' was registered.")
        else:
            print(
                f"Test accuracy ({test_accuracy:.4f}) is below threshold. Model will NOT be registered."
            )
            # We still log the model as an artifact, but we don't register it
            mlflow.xgboost.log_model(xgb_model=model, artifact_path="model")

    spark.stop()


if __name__ == "__main__":
    run_stock_xgboost_pipeline()
