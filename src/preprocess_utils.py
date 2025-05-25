import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from typing import List, Tuple
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def drop_redundant_features(
    df: DataFrame,
    threshold: float = 0.9
) -> List[str]:
    """
    Identify numeric columns to drop where |corr| > threshold.
    Returns list of column names to remove.
    """
    corr = df.corr().abs()
    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )
    to_drop = [
        col for col in upper.columns
        if any(upper[col] > threshold)
    ]
    return to_drop


def build_preprocessor(
    numerical_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
     - imputes missing numerics with median, then standard scales
     - imputes missing categoricals with most frequent, then one-hot encodes
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ], remainder='drop')
    return preprocessor


def encode_target(
    y: Series
) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Label-encode the target Series.
    Returns the encoded array and the fitted LabelEncoder.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le