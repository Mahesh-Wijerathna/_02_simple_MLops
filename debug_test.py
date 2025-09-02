#!/usr/bin/env python3

print("Starting debug test...")

try:
    import pandas as pd
    print("✓ pandas imported")
except Exception as e:
    print(f"✗ pandas import failed: {e}")

try:
    import yaml
    print("✓ yaml imported")
except Exception as e:
    print(f"✗ yaml import failed: {e}")

try:
    import sklearn
    print("✓ sklearn imported")
except Exception as e:
    print(f"✗ sklearn import failed: {e}")

try:
    import joblib
    print("✓ joblib imported")
except Exception as e:
    print(f"✗ joblib import failed: {e}")

try:
    import sys
    sys.path.append('src')
    from model import get_model
    model = get_model()
    print("✓ model imported and created")
except Exception as e:
    print(f"✗ model import failed: {e}")

try:
    df = pd.read_csv("data/data.csv")
    print(f"✓ data loaded: {len(df)} rows")
except Exception as e:
    print(f"✗ data loading failed: {e}")

try:
    with open("configs/train.yaml", "r") as f:
        config = yaml.safe_load(f)
    print(f"✓ config loaded: {config}")
except Exception as e:
    print(f"✗ config loading failed: {e}")

print("Debug test complete.")
