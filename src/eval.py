import joblib
import pandas as pd
import os
from sklearn.metrics import mean_squared_error

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "data.csv")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please train the model first.")
        return
        
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df[["area_sqft", "bedrooms", "bathrooms", "age_years"]]
    y = df["price"]

    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = mse ** 0.5  # Calculate RMSE manually
    print(f"Full-dataset RMSE={rmse:.2f}")

if __name__ == "__main__":
    main()
