from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn

import src.training.utils as U
from src.features.build_features import build_pipeline, FEATURIZATION_PARAMS

if __name__ == "__main__":

    mlflow.set_tracking_uri(U.MLFLOW_TRACKING_URI)
    print(f"Tracking URI {U.MLFLOW_TRACKING_URI}")

    data = U.load_data()

    with mlflow.start_run():

        model = build_pipeline(LinearRegression())
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

        print(f"LinearRegression: {metrics}")

        mlflow.log_param("git_commit_id", U.get_git_commit_id())
        mlflow.log_params(FEATURIZATION_PARAMS)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model, "model", registered_model_name="LinearRegression"
        )
        print("Saving model done")
