print("Testing inference...")

try:
    import joblib
    import pandas as pd
    import os
    
    MODEL_PATH = "artifacts/model.pkl"
    
    print(f"Checking if model exists: {os.path.exists(MODEL_PATH)}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        exit(1)
    
    # Load model
    model = joblib.load(MODEL_PATH)
    print("✓ Model loaded successfully")
    
    # Test sample
    test_sample = {
        "area_sqft": 1200,
        "bedrooms": 3,
        "bathrooms": 2,
        "age_years": 10
    }
    
    df = pd.DataFrame([test_sample])
    print(f"✓ Test sample prepared: {test_sample}")
    
    prediction = model.predict(df)[0]
    print(f"✓ Prediction: ${prediction:,.2f}")
    
    print("🎉 Inference completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
