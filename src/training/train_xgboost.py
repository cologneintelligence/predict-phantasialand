import json

from xgboost import XGBRegressor
import mlflow
import click

import src.training.utils as U
from src.features.build_features import build_pipeline, FEATURIZATION_PARAMS


@click.command()
@click.option(
    "-p",
    "--params",
    "params",
    default="",
    help="parameters to be passed to XGBoost as JSON string",
)
@click.option("-n", "--note", "note", default="", help="note to add to mlflow")
def main(params: str, note: str):

    if params:
        param_dict = json.loads(params)
    else:
        param_dict = {}

    mlflow.set_tracking_uri(U.MLFLOW_TRACKING_URI)
    print(f"Tracking URI {U.MLFLOW_TRACKING_URI}")

    data = U.load_data()

    with mlflow.start_run():

        model = build_pipeline(XGBRegressor(verbosity=2, random_state=42, **param_dict))
        print("Built pipeline")

        model.fit(data.X_train, data.y_train)
        print("Trained model")

        y_pred = model.predict(data.X_test)
        print("Predicted test samples")

        metrics = U.regression_metrics(data.y_test, y_pred)
        metrics.update(
            U.regression_metrics(
                data.y_train, model.predict(data.X_train), suffix="_train"
            )
        )

        print(f"XGBRegressor: {metrics}")

        mlflow.log_param("git_commit_id", U.get_git_commit_id())
        mlflow.log_param("random_state", 42)
        mlflow.log_params(FEATURIZATION_PARAMS)
        mlflow.log_params(param_dict)
        mlflow.log_metrics(metrics)

        if note:
            mlflow.set_tag("mlflow.note.content", note)

        mlflow.sklearn.log_model(model, "model", registered_model_name="XGBRegressor")
        print("Saving model done")


if __name__ == "__main__":
    main()
