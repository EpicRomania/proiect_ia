from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.preprocess_utils import build_preprocessor


def make_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str],
    random_state: int = 42
) -> Tuple[Pipeline, dict]:
    """
    Random-Forest pipeline + hyper-parameter grid.
    """
    prep = build_preprocessor(numerical_cols, categorical_cols)

    clf = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    pipeline = Pipeline([
        ("prep", prep),
        ("clf", clf),
    ])

    param_grid = {
        "clf__n_estimators": [100, 300, 500],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_leaf": [1, 2, 5],
        "clf__max_features": ["sqrt", "log2"],
        "clf__criterion": ["gini", "entropy", "log_loss"],
    }

    return pipeline, param_grid