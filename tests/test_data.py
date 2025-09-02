import pandas as pd
import os

def test_data_loads():
    assert os.path.exists("data/data.csv"), "Data file does not exist"
    df = pd.read_csv("data/data.csv")
    assert not df.empty, "Data file is empty"
    required_columns = {"area_sqft", "bedrooms", "bathrooms", "age_years", "price"}
    assert required_columns.issubset(df.columns), f"Missing columns: {required_columns - set(df.columns)}"
    assert len(df) > 0, "Data has no rows"
    
def test_data_quality():
    df = pd.read_csv("data/data.csv")
    # Check for missing values
    assert not df.isnull().any().any(), "Data contains missing values"
    # Check for reasonable ranges
    assert df["price"].min() > 0, "Price should be positive"
    assert df["area_sqft"].min() > 0, "Area should be positive"
    assert df["bedrooms"].min() > 0, "Bedrooms should be positive"
