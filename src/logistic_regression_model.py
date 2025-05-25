from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.preprocess_utils import build_preprocessor


def make_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str],
    random_state: int = 42
) -> Tuple[Pipeline, dict]:
    """
    (Regularised) Logistic-Regression pipeline + grid.
    Uses scikit-learn implementation with L2-penalty; good baseline
    while custom GD code is optional for further experiments.
    """
    prep = build_preprocessor(numerical_cols, categorical_cols)

    clf = LogisticRegression(
        random_state=random_state,
        max_iter=500,
        multi_class="auto",
        class_weight="balanced",
    )

    pipeline = Pipeline([
        ("prep", prep),
        ("clf", clf),
    ])

    param_grid = {
        "clf__solver": ["lbfgs", "liblinear", "saga"],
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
    }

    return pipeline, param_grid