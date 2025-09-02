import requests
import json

print("Testing API...")

# Test health endpoint
try:
    response = requests.get("http://localhost:8000/health")
    print(f"Health check status: {response.status_code}")
    print(f"Health response: {response.json()}")
except Exception as e:
    print(f"❌ Health check failed: {e}")

# Test prediction endpoint
try:
    test_data = {
        "area_sqft": 1200,
        "bedrooms": 3,
        "bathrooms": 2,
        "age_years": 10
    }
    
    response = requests.post(
        "http://localhost:8000/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(test_data)
    )
    
    print(f"Prediction status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Prediction: ${result['prediction']:,.2f}")
    else:
        print(f"❌ Prediction failed: {response.text}")
        
except Exception as e:
    print(f"❌ API test failed: {e}")

print("API test completed!")
