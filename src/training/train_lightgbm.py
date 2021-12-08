from lightgbm import LGBMRegressor
import mlflow
import click
import pandas as pd

import src.training.utils as U
from src.features.build_features import build_pipeline, FEATURIZATION_PARAMS


def oversample(X: pd.DataFrame, y: pd.DataFrame, factor: int, quantile: float):

    df = X.copy()
    df["y"] = y.waiting_time

    oversample_thresh = df.y.quantile(quantile)

    to_oversample = df[df.y >= oversample_thresh].copy()

    dfs = [df] + [to_oversample] * factor

    X_y_resampled = pd.concat(dfs, ignore_index=True)

    # shuffle
    X_y_resampled = X_y_resampled.sample(frac=1, random_state=42).reset_index(drop=True) 

    X_resampled = X_y_resampled.drop(columns=["y"])
    y_resampled = X_y_resampled.y

    return X_resampled, y_resampled


@click.command()
@click.option(
    "--oversample-factor",
    "oversample_factor",
    type=int,
    default=1,
    help="how often should high waiting times be sampled",
)
@click.option(
    "--oversample-quantile",
    "oversample_quantile",
    type=float,
    default=0.9,
    help="values above this quantile will be treated as high",
)
def main(oversample_factor: int, oversample_quantile: float):

    mlflow.set_tracking_uri(U.MLFLOW_TRACKING_URI)
    print(f"Tracking URI {U.MLFLOW_TRACKING_URI}")

    data = U.load_data()

    with mlflow.start_run():

        model = build_pipeline(LGBMRegressor(verbosity=2, random_state=42))
        print("Built pipeline")

        if oversample_factor > 1:
            X, y = oversample(data.X_train, data.y_train, oversample_factor, oversample_quantile)
            mlflow.log_param("oversample_factor", oversample_factor)
            mlflow.log_param("oversample_quantile", oversample_quantile)
        else:
            X, y = data.X_train, data.y_train

        model.fit(X, y.to_numpy().ravel())
        print("Trained model")

        y_pred = model.predict(data.X_test)
        print("Predicted test samples")

        metrics = U.regression_metrics(data.y_test, y_pred)
        metrics.update(
            U.regression_metrics(
                data.y_train, model.predict(data.X_train), suffix="_train"
            )
        )

        print(f"LGBMRegressor: {metrics}")

        mlflow.log_param("git_commit_id", U.get_git_commit_id())
        mlflow.log_param("random_state", 42)
        mlflow.log_params(FEATURIZATION_PARAMS)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, "model", registered_model_name="LGBMRegressor")
        print("Saving model done")


if __name__ == "__main__":
    main()
