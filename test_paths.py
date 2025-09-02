#!/usr/bin/env python3
"""
Test script to verify the path fixes work correctly
"""

import os
import sys

def test_paths():
    """Test that all scripts can find their required files"""
    print("Testing path resolution...")

    # Test train.py paths
    sys.path.append('src')
    from train import CONFIG_PATH, DATA_PATH, MODEL_PATH, METRICS_PATH

    print(f"CONFIG_PATH: {CONFIG_PATH}")
    print(f"DATA_PATH: {DATA_PATH}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"METRICS_PATH: {METRICS_PATH}")

    # Check if files exist
    checks = [
        ("Config file", CONFIG_PATH, os.path.exists(CONFIG_PATH)),
        ("Data file", DATA_PATH, os.path.exists(DATA_PATH)),
        ("Model file", MODEL_PATH, os.path.exists(MODEL_PATH)),
        ("Metrics file", METRICS_PATH, os.path.exists(METRICS_PATH)),
    ]

    all_good = True
    for name, path, exists in checks:
        status = "✅ EXISTS" if exists else "❌ MISSING"
        print(f"{name}: {status}")
        if not exists:
            all_good = False

    return all_good

def test_imports():
    """Test that all imports work correctly"""
    print("\nTesting imports...")

    try:
        from train import main as train_main
        print("✅ train.py import: SUCCESS")
    except Exception as e:
        print(f"❌ train.py import: FAILED - {e}")
        return False

    try:
        from eval import main as eval_main
        print("✅ eval.py import: SUCCESS")
    except Exception as e:
        print(f"❌ eval.py import: FAILED - {e}")
        return False

    try:
        from infer import predict
        print("✅ infer.py import: SUCCESS")
    except Exception as e:
        print(f"❌ infer.py import: FAILED - {e}")
        return False

    return True

if __name__ == "__main__":
    print("🔍 Path Resolution Test")
    print("=" * 40)

    paths_ok = test_paths()
    imports_ok = test_imports()

    print("\n" + "=" * 40)
    if paths_ok and imports_ok:
        print("🎉 ALL TESTS PASSED! Path fixes are working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Check the output above.")
        sys.exit(1)
