from typing import Dict, Union
import numpy as np
import optuna
from xgboost import XGBRegressor
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from sklearn.model_selection import cross_validate, KFold

import src.training.utils as U
from src.features.build_features import build_pipeline, FEATURIZATION_PARAMS


def _log_params(client, id, dict_, prefix=""):

    for k, v in dict_.items():
        client.log_param(id, f"{prefix}{k}", v)


def _log_metrics(client, id, dict_):

    for k, v in dict_.items():
        client.log_metric(id, k, v)


# adapted from:
# https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_cv_integration.py
def setup_param_dict(trial) -> Dict[str, Union[str, float]]:
    """Choose the parameters for the current trial based on optuna's suggestions.

    TODO adjust parameters and their range of values

    Args:
        trial: current trial

    Returns:
        dict: XGBoost parameters
    """

    param = {
        "verbosity": 1,
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 10, 800),
        "max_depth": trial.suggest_int("max_depth", 5, 50),
        "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.2),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
    }

    return param


# adapted from:
# https://simonhessner.de/mlflow-optuna-parallel-hyper-parameter-optimization-and-logging/
def get_objective(parent_run_id):
    """function to create a objective function for optuna tuning. The nested functions
    are needed to supply the mlflow context correctly.

    Args:
        parent_run_id: id of mlflow parent run

    Returns:
        objective function for MLFlow tuning
    """
    # get an objective function for optuna that creates nested MLFlow runs

    def objective(trial):
        trial_run = client.create_run(
            experiment_id=experiment, tags={MLFLOW_PARENT_RUN_ID: parent_run_id}
        )

        data = U.load_data()

        param = setup_param_dict(trial)

        model = build_pipeline(XGBRegressor(**param, random_state=42))

        cv_results = cross_validate(
            model,
            data.X_train,
            data.y_train,
            scoring=("neg_root_mean_squared_error", "r2", "neg_mean_absolute_error"),
            cv=KFold(3, shuffle=False),
            return_train_score=True,
        )

        metrics = {
            "r2": np.mean(cv_results["test_r2"]),
            "rmse": -np.mean(cv_results["test_neg_root_mean_squared_error"]),
            "mae": -np.mean(cv_results["test_neg_mean_absolute_error"]),
            "r2_train": np.mean(cv_results["train_r2"]),
            "rmse_train": -np.mean(cv_results["train_neg_root_mean_squared_error"]),
            "mae_train": -np.mean(cv_results["train_neg_mean_absolute_error"]),
        }

        _log_params(client, trial_run.info.run_id, param)
        _log_params(
            client,
            trial_run.info.run_id,
            {
                "random_state": 42,
                "cv_splits": 3,
                "cv_shuffle": False,
                "git_commit_id": U.get_git_commit_id(),
            },
        )
        _log_params(client, trial_run.info.run_id, FEATURIZATION_PARAMS)

        _log_metrics(client, trial_run.info.run_id, metrics)

        return metrics["rmse"]

    return objective


if __name__ == "__main__":

    client = MlflowClient(U.MLFLOW_TRACKING_URI)
    experiment_name = "min_rmse_XGBRegressor2"
    try:
        experiment = client.create_experiment(experiment_name)
    except:
        experiment = client.get_experiment_by_name(experiment_name).experiment_id

    study_run = client.create_run(experiment_id=experiment)
    study_run_id = study_run.info.run_id

    study = optuna.create_study(
        direction="minimize", study_name=experiment_name, storage="sqlite:///optuna.db"
    )
    study.optimize(get_objective(study_run_id), n_trials=100, show_progress_bar=True)

    _log_params(client, study_run_id, study.best_trial.params)
    client.log_metric(study_run_id, "rmse", study.best_value)
