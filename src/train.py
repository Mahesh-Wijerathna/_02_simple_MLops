import pandas as pd
import yaml
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from model import get_model

# Paths relative to project root (go up one level from src/)
CONFIG_PATH = "../configs/train.yaml"
DATA_PATH = "../data/data.csv"
MODEL_PATH = "../artifacts/model.pkl"
METRICS_PATH = "../artifacts/metrics.txt"

def main():
    # Create artifacts directory if it doesn't exist
    os.makedirs("../artifacts", exist_ok=True)
    
    # load config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(DATA_PATH)
    X = df[["area_sqft", "bedrooms", "bathrooms", "age_years"]]
    y = df["price"]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"]
    )

    # train
    model = get_model()
    model.fit(X_train, y_train)

    # eval
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5  # Calculate RMSE manually

    # save
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        f.write(f"RMSE: {rmse:.2f}\n")

    print(f"Training complete. RMSE={rmse:.2f}")

if __name__ == "__main__":
    main()
