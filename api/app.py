import sys
import os

# Ensure project root directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import io

from src.data_preprocessing import create_features
from src.db import log_prediction, get_all_predictions

app = FastAPI(
    title="Medical Cost Predictor API",
    description="Production REST API for Medical Charges prediction with Quantile Loss confidence intervals & DB history",
    version="1.0.0"
)

# Global model pointers
model = None
preprocessor = None
q_low = None
q_high = None

@app.on_event("startup")
def load_artifacts():
    global model, preprocessor, q_low, q_high
    model_path = "models/best_model.pkl"
    prep_path = "models/preprocessor.pkl"
    q_low_path = "models/quantile_low.pkl"
    q_high_path = "models/quantile_high.pkl"

    if os.path.exists(model_path) and os.path.exists(prep_path):
        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)
    if os.path.exists(q_low_path) and os.path.exists(q_high_path):
        q_low = joblib.load(q_low_path)
        q_high = joblib.load(q_high_path)

load_artifacts()

class PatientSchema(BaseModel):
    age: int = Field(..., ge=18, le=100, json_schema_extra={"example": 35})
    sex: str = Field(..., pattern="^(male|female)$", json_schema_extra={"example": "female"})
    bmi: float = Field(..., ge=10.0, le=60.0, json_schema_extra={"example": 28.5})
    children: int = Field(..., ge=0, le=10, json_schema_extra={"example": 1})
    smoker: str = Field(..., pattern="^(yes|no)$", json_schema_extra={"example": "no"})
    region: str = Field(..., pattern="^(southeast|southwest|northeast|northwest)$", json_schema_extra={"example": "southwest"})

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "quantile_models_loaded": q_low is not None and q_high is not None,
    }

@app.post("/predict")
def predict_single(patient: PatientSchema):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded. Run model training first.")

    input_df = pd.DataFrame([patient.dict()])
    engineered_df = create_features(input_df)

    processed = preprocessor.transform(engineered_df)
    log_pred = float(model.predict(processed)[0])
    pred_cost = float(np.expm1(log_pred))

    # Quantile bounds
    if q_low is not None and q_high is not None:
        low_bound = float(np.expm1(q_low.predict(processed)[0]))
        high_bound = float(np.expm1(q_high.predict(processed)[0]))
    else:
        low_bound = pred_cost * 0.85
        high_bound = pred_cost * 1.15

    # Log to SQLite DB
    log_prediction(patient.dict(), pred_cost, low_bound, high_bound)

    return {
        "patient": patient.dict(),
        "predicted_cost_usd": round(pred_cost, 2),
        "confidence_interval_usd": {
            "low_10th_percentile": round(low_bound, 2),
            "median_prediction": round(pred_cost, 2),
            "high_90th_percentile": round(high_bound, 2),
        }
    }

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded.")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    engineered_df = create_features(df)

    processed = preprocessor.transform(engineered_df)
    log_preds = model.predict(processed)
    df["predicted_cost_usd"] = np.round(np.expm1(log_preds), 2)

    return df.to_dict(orient="records")

@app.get("/history")
def fetch_history(limit: int = 50):
    return get_all_predictions(limit=limit)
