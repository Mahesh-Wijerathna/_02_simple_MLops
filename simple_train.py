import pandas as pd
import yaml
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def main():
    print("Starting training...")
    
    # Create artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)
    print("Created artifacts directory")
    
    # load config
    with open("configs/train.yaml", "r") as f:
        config = yaml.safe_load(f)
    print(f"Loaded config: {config}")

    df = pd.read_csv("data/data.csv")
    print(f"Loaded data: {len(df)} rows")
    
    X = df[["area_sqft", "bedrooms", "bathrooms", "age_years"]]
    y = df["price"]
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"]
    )
    print(f"Split data - Train: {X_train.shape}, Test: {X_test.shape}")

    # train
    model = LinearRegression()
    print("Created model")
    
    model.fit(X_train, y_train)
    print("Trained model")

    # eval
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5  # Calculate RMSE manually
    print(f"RMSE: {rmse:.2f}")

    # save
    MODEL_PATH = "artifacts/model.pkl"
    METRICS_PATH = "artifacts/metrics.txt"
    
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    
    with open(METRICS_PATH, "w") as f:
        f.write(f"RMSE: {rmse:.2f}\n")
    print(f"Saved metrics to {METRICS_PATH}")

    print(f"Training complete. RMSE={rmse:.2f}")

if __name__ == "__main__":
    main()
