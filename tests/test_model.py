import joblib
import pandas as pd
import os
import sys
sys.path.append('src')
from train import main as train_main

def test_model_predicts():
    # Train model first if it doesn't exist
    if not os.path.exists("artifacts/model.pkl"):
        train_main()
    
    model = joblib.load("artifacts/model.pkl")
    sample = pd.DataFrame([{
        "area_sqft": 1000,
        "bedrooms": 2,
        "bathrooms": 1,
        "age_years": 5
    }])
    pred = model.predict(sample)
    assert pred.shape[0] == 1
    assert isinstance(pred[0], (int, float))
