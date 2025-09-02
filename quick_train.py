import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Create artifacts directory
os.makedirs("artifacts", exist_ok=True)

# Load and prepare data
df = pd.read_csv("data/data.csv")
X = df[["area_sqft", "bedrooms", "bathrooms", "age_years"]]
y = df["price"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
rmse = mse ** 0.5  # Calculate RMSE manually

# Save
joblib.dump(model, "artifacts/model.pkl")

# Write results to file
with open("artifacts/training_results.txt", "w") as f:
    f.write(f"Training completed successfully!\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"Model saved to artifacts/model.pkl\n")
    f.write(f"Training samples: {X_train.shape[0]}\n")
    f.write(f"Test samples: {X_test.shape[0]}\n")

# Also create a simple verification file
with open("training_completed.txt", "w") as f:
    f.write(f"SUCCESS - RMSE: {rmse:.2f}")
