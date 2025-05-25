import pandas as pd
from pandas import DataFrame
from typing import Tuple, List

def load_data(path):
    """
    Load a CSV file into a pandas DataFrame.
    """
    return pd.read_csv(path)

def split_features_target(df, target_col):
    """
    Split DataFrame into features X and target y.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    return X, y

def inspect_dataframe(df):
    """
    Print shape, dtypes, and missing‐value counts.
    """
    print('Shape:', df.shape)
    print('Dtypes:')
    print(df.dtypes)
    print('Missing values:')
    print(df.isna().sum())

def get_column_types(df: DataFrame,cat_threshold: int = 10) -> Tuple[List[str], List[str]]:
    """
    Return lists of numerical and categorical column names.
    Treat any numeric column with <= cat_threshold unique values as categorical.
    """
    # all numeric columns
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # pick out low-cardinality numerics as categorical
    low_card = [col for col in num_cols if df[col].nunique() <= cat_threshold]
    # remaining true numerics
    true_num = [col for col in num_cols if col not in low_card]
    # include any object/string columns as categorical
    obj_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    cat_cols = low_card + obj_cols
    return true_num, cat_cols

