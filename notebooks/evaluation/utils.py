from typing import List

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import plotly.graph_objects as go
import joblib
from plotly.subplots import make_subplots

from src.features.build_features import FEATURE_MATRIX_COLUMNS
from src.training.utils import *


def train_save(model, save_path, data):

    model.fit(data.X_train_p, data.y_train)

    joblib.dump(model, str(save_path) + f"_{get_git_commit_id()}.joblib")


def load_predict_model(path, data):

    model = joblib.load(path)
    y_pred_test = model.predict(data.X_test_p)
    y_pred_train = model.predict(data.X_train_p)

    test_df = pd.DataFrame(
        {
            "date": data.X_test.date,
            "time": data.X_test.half_hour_time,
            "attraction": data.X_test.attraction,
            "y_true": data.y_test.to_numpy().flatten(),
            "y_pred": y_pred_test.flatten(),
        }
    )

    train_df = pd.DataFrame(
        {
            "date": data.X_train.date,
            "time": data.X_train.half_hour_time,
            "attraction": data.X_train.attraction,
            "y_true": data.y_train.to_numpy().flatten(),
            "y_pred": y_pred_train.flatten(),
        }
    )

    return model, test_df, train_df


def regression_scatter_plot(
    result_df, attractions: List[str], model_name, col_wrap=4, height=1200, width=1200
):

    df = result_df.query("attraction in @attractions")

    min_ = min(df.y_true.min(), df.y_pred.min())
    max_ = max(df.y_true.max(), df.y_pred.max())

    rows = int(np.ceil(len(attractions) / col_wrap))
    cols = min(len(attractions), col_wrap)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes="all",
        shared_yaxes="all",
        subplot_titles=attractions,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        specs=[[{}] * cols] * rows,
    )

    for idx, attraction_name in enumerate(attractions):

        attraction_df = df.query("attraction == @attraction_name")

        # Who the hell starts counting at 1?!
        row = idx // col_wrap + 1
        col = idx % col_wrap + 1

        fig.add_trace(
            go.Scatter(
                x=attraction_df.y_true,
                y=attraction_df.y_pred,
                mode="markers",
                name="observations",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[min_ - 5, max_ + 5],
                y=[min_ - 5, max_ + 5],
                mode="lines",
                name="goal line",
            ),
            row=row,
            col=col,
        )

        regline_x_values = attraction_df.y_true.to_numpy().reshape(-1, 1)

        regress_line = (
            LinearRegression()
            .fit(regline_x_values, attraction_df.y_pred)
            .predict(regline_x_values)
        )

        fig.add_trace(
            go.Scatter(
                x=attraction_df.y_true,
                y=regress_line,
                mode="lines",
                name="regress line",
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        height=height,
        width=width,
        title_text=f"predictions vs actual observations for {model_name}",
        xaxis_title="ground truth (min)",
        yaxis_title="prediction (min)",
        showlegend=False,
    )

    fig.show()


def get_xgb_fscores(xgb_model):
    fscores = xgb_model.get_booster().get_fscore()
    fscore_df = pd.DataFrame(fscores.values(), index=fscores.keys(), columns=["fscore"])
    fscore_df.index = fscore_df.index.map(lambda x: int(x.strip("f")))
    name_s = pd.Series(FEATURE_MATRIX_COLUMNS)
    name_s.name = "feature_name"
    fscore_df = fscore_df.join(other=name_s)
    fscore_df.sort_values(by="fscore", ascending=True, inplace=True)
    return fscore_df


def regression_metrics(y_true, y_pred, suffix=""):

    return {
        f"r2{suffix}": metrics.r2_score(y_true, y_pred),
        # TODO r2 adjusted
        f"rmse{suffix}": metrics.mean_squared_error(y_true, y_pred, squared=False),
        f"mae{suffix}": metrics.mean_absolute_error(y_true, y_pred),
    }
