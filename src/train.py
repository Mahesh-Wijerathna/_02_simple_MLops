import pandas as pd
import yaml
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from model import get_model

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Paths relative to project root
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "train.yaml")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "data.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")
METRICS_PATH = os.path.join(PROJECT_ROOT, "artifacts", "metrics.txt")

def main():
    # Create artifacts directory if it doesn't exist
    os.makedirs(os.path.join(PROJECT_ROOT, "artifacts"), exist_ok=True)
    
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
