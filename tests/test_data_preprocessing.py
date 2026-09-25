import pandas as pd
import numpy as np
from src.data_preprocessing import create_features, build_preprocessor

def test_create_features_bmi_smoker_interaction():
    df = pd.DataFrame({
        "age": [25, 40],
        "sex": ["male", "female"],
        "bmi": [30.0, 25.0],
        "children": [0, 1],
        "smoker": ["yes", "no"],
        "region": ["southwest", "southeast"],
        "charges": [10000.0, 5000.0]
    })
    feat_df = create_features(df)
    assert "bmi_smoker" in feat_df.columns
    assert "bmi_sq" in feat_df.columns
    assert "age_sq" in feat_df.columns
    assert "age_smoker" in feat_df.columns
    assert "bmi_risk_tier" in feat_df.columns
    assert feat_df["bmi_smoker"].iloc[0] == 30.0  # Smoker: 30 * 1
    assert feat_df["bmi_smoker"].iloc[1] == 0.0   # Non-smoker: 25 * 0

def test_preprocessor_transform_shape():
    raw_df = pd.DataFrame({
        "age": [30],
        "sex": ["female"],
        "bmi": [28.0],
        "children": [1],
        "smoker": ["no"],
        "region": ["northwest"],
        "charges": [4000.0]
    })
    feat_df = create_features(raw_df)
    X = feat_df.drop(columns=["charges"])
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == 1
    assert transformed.shape[1] > 0
