
import datetime
import itertools
from typing import Any, Tuple
import numpy as np
from pathlib import Path

import pandas as pd
import mlflow
from src.models.base import WeatherBinEstimator

from src.training.utils import MLFLOW_TRACKING_URI
from src.models.weather_bins import Bin, get_weather_data_for_bin, ALL_WEATHER_BINS

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

BEST_MODEL_PATH = (Path(__file__).parent.parent.parent / "models" / "best").resolve()

TIMES = pd.Series(
    itertools.chain.from_iterable(
        [f"{hour}:00:00", f"{hour}:30:00"] for hour in range(10, 20)
    )
)


def generate_X(
    date: datetime.date, attraction: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate all datapoints for which we need to request the model.

    This is the cartesian product of TIMES and `get_weather_data_for_bin`, with data and
    attraction always being fixed.

    Args:
        date (datetime.date): date for which to query the model
        attraction (str): attraction for which to query the model

    Returns:
        X (pd.DataFrame): feature matrix for prediction (columns: "attraction", "date",
            "half_hour_time" and all weather columns).
        bins_time (pd.DataFrame): information needed for correct summarization of the
            prediction, same number of rows as `X` (columns: "half_hour_time",
            DRY_SUNNY, DRY_OVERCAST, SLIGHT_RAIN, HEAVY_RAIN).
    """

    weather_bins_df = get_weather_data_for_bin(date.month)

    df = pd.DataFrame(
        {"attraction": attraction, "date": date.isoformat(), "half_hour_time": TIMES}
    )

    X_with_bins = pd.merge(df, weather_bins_df, how="cross")

    bins_time = X_with_bins[[*ALL_WEATHER_BINS, "half_hour_time"]]
    X = X_with_bins.drop(columns=ALL_WEATHER_BINS)

    return X, bins_time


def summarize_waiting_times(
    y: pd.Series, bins_time: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize all model prediction to get a meaningful forecast.

    This creates two results:
    - A waiting time prediction for each weather bin and time by taking the median of
        the corresponding predictions.
    - A summary of the day containing the mean waiting time (averaged over all half-hour
        points in time) and the support for each weather bin (i.e. the number of days
        with that weather used for the prediction).

    Args:
        y (pd.Series): model prediction
        bins_time (pd.DataFrame): descriptor generated by `generate_X`, same number of
            rows as `y`

    Returns:
        pd.DataFrame: median waiting time. columns: weather bins including ALL, rows:
            `TIMES`
        pd.DataFrame: daily summary. rows: weather bins including ALL, columns:
            "mean_waiting_time", "support", "best_time"
    """

    # TODO confidence values

    bins_time["y"] = y

    waiting_time_by_weather = {}

    support_by_weather = {}

    for bin in ALL_WEATHER_BINS:
        waiting_time_by_weather[bin] = (
            bins_time[bins_time[bin]][["half_hour_time", "y"]]
            .groupby(by="half_hour_time")
            .median()["y"]
        )
        # We divide by len(TIMES) to "undo" the cross-product
        support_by_weather[bin] = bins_time[bin].value_counts()[True] / len(TIMES)

    waiting_time_by_weather[Bin.ALL] = (
        bins_time[["half_hour_time", "y"]].groupby(by="half_hour_time").median()["y"]
    )
    support_by_weather[Bin.ALL] = len(bins_time) / len(TIMES)

    waiting_time_by_weather_df = pd.DataFrame(waiting_time_by_weather)

    daily_summary_df = pd.DataFrame(
        {
            "mean_waiting_time": waiting_time_by_weather_df.mean(axis="index"),
            "support": support_by_weather,
            "best_time": (
                waiting_time_by_weather_df.idxmin(axis="index")
                if not waiting_time_by_weather_df.empty
                else np.nan
            ),
        }
    )

    return waiting_time_by_weather_df, daily_summary_df


class ModelEstimator(WeatherBinEstimator):
    """Prediction model based on using a machine learning model to get a waiting time 
    estimate for a specific attraction, date, time and weather. As weather data for the 
    future is not available, it is approximated by similar weather data from the past.
    """

    model: Any = None

    def __init__(self, model_uri: str):

        self.model = mlflow.sklearn.load_model(model_uri)

    def predict(self, date: datetime.date, attraction: str) -> pd.DataFrame:
        """Predict expected waiting times for `date` and `attraction`.

        This creates two results:
        - A waiting time prediction for each weather bin and time by taking the median 
            of the corresponding predictions.
        - A summary of the day containing the mean waiting time (averaged over all 
            half-hour points in time), the support for each weather bin (i.e. the 
            number of days with that weather used for the prediction) and the time of 
            minimal waiting time for each weather bin.

        Args:
            date (datetime.date): date for which to query the model
            attraction (str): attraction for which to query the model

        Returns:
            pd.DataFrame: median waiting time. columns: weather bins including ALL;
                rows: `TIMES`
            pd.DataFrame: daily summary. columns: "mean_waiting_time", "support", 
                "best_time"; rows: weather bins including ALL
        """

        X, bins_time = generate_X(date, attraction)
        y = self.model.predict(X)
        waiting_time_by_weather_df, daily_summary_df = summarize_waiting_times(
            y, bins_time
        )

        return waiting_time_by_weather_df, daily_summary_df
