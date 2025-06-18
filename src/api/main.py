import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import numpy as np
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator


class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictionResponse(BaseModel):
    prediction: int



model = None
MODEL_NAME = "iris_xgboost_classifier"
MODEL_STAGE = "Production"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This block of code runs ONCE, on startup
    global model
    print("--- Running application startup logic ---")

    # Configure MLflow client
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        print(f"Connected to MLflow Tracking Server at {mlflow_tracking_uri}")
    else:
        print("Warning: MLFLOW_TRACKING_URI not set.")

    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

    try:
        print(f"Loading model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")
    except mlflow.exceptions.MlflowException as e:
        print(f"Error loading model: {e}")
        print("API will start without a loaded model.")

    # The 'yield' statement separates startup logic from shutdown logic
    yield

    # This block of code runs ONCE, on shutdown (not used here, but good practice)
    print("--- Running application shutdown logic ---")
    model = None

app = FastAPI(
    title="Iris Model API",
    description="API for serving the Iris XGBoost classifier.",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)

# --- API Endpoints ---
@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: IrisRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please ensure a model has been promoted to 'Production'.",
        )

    input_data = pd.DataFrame([request.model_dump()])

    try:
        # 1. The model predicts probabilities for each class, e.g., [[0.1, 0.8, 0.1]]
        prediction_probabilities = model.predict(input_data)

        # 2. Use np.argmax() to find the index of the highest probability.
        #    This index corresponds to the predicted class label.
        #    We take the first element [0] because we are predicting on a single row.
        predicted_class_index = np.argmax(prediction_probabilities[0])

        # 3. Convert the result to a standard Python int for the JSON response.
        prediction_value = int(predicted_class_index)

        return {"prediction": prediction_value}
        # ---

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during prediction: {str(e)}"
        )
