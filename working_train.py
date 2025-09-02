print("Starting simple training...")

try:
    import pandas as pd
    print("✓ pandas imported")
    
    import joblib
    print("✓ joblib imported")
    
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    print("✓ sklearn imported")
    
    import os
    
    # Create artifacts directory
    os.makedirs("artifacts", exist_ok=True)
    print("✓ artifacts directory created")
    
    # Load data
    df = pd.read_csv("data/data.csv")
    print(f"✓ data loaded: {len(df)} rows")
    
    # Prepare features
    X = df[["area_sqft", "bedrooms", "bathrooms", "age_years"]]
    y = df["price"]
    print(f"✓ features prepared: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✓ data split: train={X_train.shape[0]}, test={X_test.shape[0]}")
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✓ model trained")
    
    # Evaluate
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5  # Calculate RMSE manually
    print(f"✓ evaluation complete: RMSE={rmse:.2f}")
    
    # Save model
    joblib.dump(model, "artifacts/model.pkl")
    print("✓ model saved to artifacts/model.pkl")
    
    # Save metrics
    with open("artifacts/metrics.txt", "w") as f:
        f.write(f"RMSE: {rmse:.2f}\n")
    print("✓ metrics saved")
    
    print(f"\n🎉 Training completed successfully! RMSE: {rmse:.2f}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
