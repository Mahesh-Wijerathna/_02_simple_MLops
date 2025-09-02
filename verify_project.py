print("🔍 Project Verification Summary")
print("=" * 40)

# Check data
try:
    import pandas as pd
    df = pd.read_csv("data/data.csv")
    print(f"✅ Data: {len(df)} rows loaded")
except Exception as e:
    print(f"❌ Data: {e}")

# Check model exists
import os
if os.path.exists("artifacts/model.pkl"):
    print("✅ Model: Trained model exists")
else:
    print("❌ Model: No trained model found")

# Check metrics
if os.path.exists("artifacts/metrics.txt"):
    with open("artifacts/metrics.txt", "r") as f:
        metrics = f.read().strip()
    print(f"✅ Metrics: {metrics}")
else:
    print("❌ Metrics: No metrics file found")

# Test inference
try:
    import joblib
    model = joblib.load("artifacts/model.pkl")
    test_data = pd.DataFrame([{
        "area_sqft": 1200,
        "bedrooms": 3,
        "bathrooms": 2,
        "age_years": 10
    }])
    prediction = model.predict(test_data)[0]
    print(f"✅ Inference: Sample prediction = ${prediction:,.2f}")
except Exception as e:
    print(f"❌ Inference: {e}")

# Check API code
if os.path.exists("api/app.py"):
    print("✅ API: FastAPI app available")
else:
    print("❌ API: No API file found")

# Check tests
if os.path.exists("tests/test_data.py") and os.path.exists("tests/test_model.py"):
    print("✅ Tests: Unit tests available")
else:
    print("❌ Tests: Missing test files")

print("\n🏁 Verification Complete!")
print("\nTo run the API server:")
print("  python.exe -m uvicorn api.app:app --reload")
print("\nTo test API (in another terminal):")
print("  curl -X POST http://localhost:8000/predict -H \"Content-Type: application/json\" -d '{\"area_sqft\": 1200, \"bedrooms\": 3, \"bathrooms\": 2, \"age_years\": 10}'")
