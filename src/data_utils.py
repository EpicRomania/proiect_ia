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

def inspect_dataframe(df: DataFrame) -> DataFrame:
    """
    Return a summary table with columns:
      - Variables
      - Type
      - Missing Values
    """
    info = pd.DataFrame({
        "Variables": df.columns.tolist(),
        "Type": df.dtypes.astype(str).tolist(),
        "Missing Values": df.isna().sum().tolist()
    })
    print(info.to_string(index=False))
    return info

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

