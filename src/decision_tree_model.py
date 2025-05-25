from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocess_utils import build_preprocessor


def make_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str],
    random_state: int = 42
) -> Tuple[Pipeline, dict]:
    """
    Decision-Tree pipeline + hyper-parameter grid.
    """
    prep = build_preprocessor(numerical_cols, categorical_cols)

    clf = DecisionTreeClassifier(
        random_state=random_state,
        class_weight="balanced",   # helps with any class imbalance
    )

    pipeline = Pipeline([
        ("prep", prep),
        ("clf", clf),
    ])

    param_grid = {
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 5, 10],
        "clf__criterion": ["gini", "entropy", "log_loss"],
    }

    return pipeline, param_grid