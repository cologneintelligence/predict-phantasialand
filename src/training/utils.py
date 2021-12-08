import subprocess
from types import SimpleNamespace
from typing import Dict, Iterable
from os import PathLike
from pathlib import Path

import pandas as pd
from sklearn import metrics

from src.data.constants import DATA_PATH


_MLFLOW_DB_PATH = (Path(__file__).parent.parent.parent / "mlflow.db").resolve()

MLFLOW_TRACKING_URI = f"sqlite:///{_MLFLOW_DB_PATH}"


def load_data(path: PathLike = None) -> SimpleNamespace:
    """load training and test data from the given path.

    `path` must contain the files "(X|y)_(train|test).csv".

    Args:
        path (PathLike): where to find the training data

    Returns:
        SimpleNamespace: plain objects with the attributes (X|y)_(train|test).
    """
    if path is None:
        path = DATA_PATH / "processed"

    data = SimpleNamespace()

    for matrix in ["X_train", "X_test", "y_train", "y_test"]:
        setattr(data, matrix, pd.read_csv(f"{path}/{matrix}.csv"))

    return data


def get_git_commit_id() -> str:
    """return the git commit hash of HEAD.

    Returns:
        str: git commit hash
    """

    rev_parse_out = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )

    rev_parse_out.check_returncode()

    return rev_parse_out.stdout.strip()


def regression_metrics(
    y_true: Iterable, y_pred: Iterable, suffix: str = ""
) -> Dict[str, float]:
    """Calculate r2 score, root mean squared error and mean absolute error.

    Args:
        y_true (Iterable): true values
        y_pred (Iterable): prediction
        suffix (str, optional): optional suffix for the keys in the return dict.
            Defaults to "".

    Returns:
        Dict[str, float]: dictionary of metrics. keys: r2, rmse, mae.
    """

    return {
        f"r2{suffix}": metrics.r2_score(y_true, y_pred),
        f"rmse{suffix}": metrics.mean_squared_error(y_true, y_pred, squared=False),
        f"mae{suffix}": metrics.mean_absolute_error(y_true, y_pred),
    }
