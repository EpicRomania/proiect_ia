from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

from src.preprocess_utils import build_preprocessor


def make_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str],
    random_state: int = 42
) -> Tuple[Pipeline, dict]:
    """
    MLP-Classifier pipeline + hyper-parameter grid.
    """
    prep = build_preprocessor(numerical_cols, categorical_cols)

    clf = MLPClassifier(
        random_state=random_state,
        max_iter=200,
        early_stopping=True,
    )

    pipeline = Pipeline([
        ("prep", prep),
        ("clf", clf),
    ])

    param_grid = {
        "clf__hidden_layer_sizes": [(64,), (128,), (64, 32)],
        "clf__activation": ["relu", "tanh"],
        "clf__learning_rate_init": [1e-3, 1e-4],
        "clf__alpha": [1e-4, 1e-3],        # L2 regularisation
        "clf__batch_size": [64, 128],
    }

    return pipeline, param_grid