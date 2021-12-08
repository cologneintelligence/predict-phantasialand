from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn
import numpy as np

import src.training.utils as U
from src.features.build_features import build_pipeline, FEATURIZATION_PARAMS

if __name__ == "__main__":

    mlflow.set_tracking_uri(U.MLFLOW_TRACKING_URI)
    print(f"Tracking URI {U.MLFLOW_TRACKING_URI}")

    data = U.load_data()

    with mlflow.start_run():

        model = build_pipeline(LinearRegression())
        print("Built pipeline")

        y_train_log = np.log(data.y_train + 1)

        model.fit(data.X_train, y_train_log)
        print("Trained model")

        y_pred_log = model.predict(data.X_test)
        y_pred = np.exp(y_pred_log) - 1

        y_pred_train_log = model.predict(data.X_train)
        y_pred_train = np.exp(y_pred_train_log) - 1
        print("Predicted test samples")

        metrics = U.regression_metrics(data.y_test, y_pred)
        metrics.update(
            U.regression_metrics(data.y_train, y_pred_train, suffix="_train")
        )

        print(f"LinearRegression with log: {metrics}")

        mlflow.log_param("git_commit_id", U.get_git_commit_id())
        mlflow.log_params(FEATURIZATION_PARAMS)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model, "model", registered_model_name="LogLinearRegression"
        )
        print("Saving model done")
