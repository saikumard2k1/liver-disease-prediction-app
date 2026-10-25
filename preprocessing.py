"""Shared preprocessing: must mirror the notebook exactly.

Notebook steps applied before scaling:
1. Gender: Male -> 1, Female -> 0
2. Albumin and Globulin: missing values filled with the training median
3. log1p on the five right-skewed features
"""
import numpy as np
import pandas as pd

FEATURES = [
    "Age", "Gender", "Total Bilirubin", "Direct Bilirubin",
    "Alkaline Phosphotase", "Alamine Aminotransferase",
    "Aspartate Aminotransferase", "Total Protiens", "Albumin",
    "Albumin and Globulin",
]

SKEWED = [
    "Total Bilirubin", "Direct Bilirubin", "Alamine Aminotransferase",
    "Aspartate Aminotransferase", "Alkaline Phosphotase",
]


def prepare_input(df_raw: pd.DataFrame, ag_median: float = 0.95) -> pd.DataFrame:
    """Turn raw patient rows into the feature frame the scaler/model expects."""
    X = df_raw[FEATURES].copy()

    if not pd.api.types.is_numeric_dtype(X["Gender"]):
        X["Gender"] = X["Gender"].astype(str).str.strip().str.title().map({"Male": 1, "Female": 0})

    X["Albumin and Globulin"] = X["Albumin and Globulin"].fillna(ag_median)

    for col in SKEWED:
        X[col] = np.log1p(X[col])

    return X
