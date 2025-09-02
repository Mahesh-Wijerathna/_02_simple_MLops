import joblib
import pandas as pd
import os

MODEL_PATH = "../artifacts/model.pkl"

def predict(sample: dict):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
    
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([sample])
    return model.predict(df)[0]

if __name__ == "__main__":
    test_sample = {
        "area_sqft": 1200,
        "bedrooms": 3,
        "bathrooms": 2,
        "age_years": 10
    }
    try:
        prediction = predict(test_sample)
        print("Prediction:", prediction)
    except FileNotFoundError as e:
        print(f"Error: {e}")
