import datetime
import numpy as np

import pandas as pd

from src.models.base import WeatherBinEstimator
from src.models.weather_bins import ALL_WEATHER_BINS, Bin, get_weather_data_for_bin
from src.training.utils import load_data

class MeanEstimator(WeatherBinEstimator):
    """Prediction model based on averaging all datapoints in the training set from the 
    same month and with similar weather as the request.
    """

    data_df: pd.DataFrame = None

    def __init__(self):

        data = load_data()

        data.X_train["waiting_time"] = data.y_train
        data.X_train["date"] = pd.to_datetime(data.X_train.date)

        self.data_df = data.X_train

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

        bin_data = get_weather_data_for_bin(date.month)
        bin_data[Bin.ALL] = True  # This allows for treating ALL like any other bin

        waiting_time_by_weather = {}
        support_by_weather = {}

        for bin in [*ALL_WEATHER_BINS, Bin.ALL]:

            relevant_dates = bin_data[bin_data[bin]].index.intersection(
                pd.to_datetime(self.data_df.date)
            )

            support_by_weather[bin] = len(relevant_dates)

            support_rows = self.data_df.query(
                "date in @relevant_dates and attraction == @attraction"
            )

            waiting_time_by_weather[bin] = (
                support_rows[["half_hour_time", "waiting_time"]]
                .groupby(by="half_hour_time")
                .mean()["waiting_time"]
            )

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
