from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

MODEL_PATH = "artifacts/model.pkl"

class House(BaseModel):
    area_sqft: float
    bedrooms: int
    bathrooms: int
    age_years: int

app = FastAPI()

def load_model():
    """Load model with error handling"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
    return joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(house: House):
    try:
        model = load_model()
        df = pd.DataFrame([house.dict()])
        pred = model.predict(df)[0]
        return {"prediction": float(pred)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
def health_check():
    model_exists = os.path.exists(MODEL_PATH)
    return {"status": "healthy", "model_loaded": model_exists}
