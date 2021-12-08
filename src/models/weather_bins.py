import pandas as pd
import numpy as np

from src.data.create_training_data import prepare_weather
from src.data.constants import DATA_PATH
from src.training.utils import load_data


def _load_weather_df():
    """Load a merged version of all weather data from Lommersum and Koeln-Bonn.

    Returns:
        pd.DataFrame: DataFrame containing merged weather data as in the training data
    """

    lommersum_df = pd.read_csv(
        DATA_PATH / "interim/weather_station01327_Lommersum.csv",
        index_col="date",
        parse_dates=["date"],
    )
    koelnbonn_df = pd.read_csv(
        DATA_PATH / "interim/weather_station02667_Koeln-Bonn.csv",
        index_col="date",
        parse_dates=["date"],
    )

    training_data = load_data()
    test_dates = pd.to_datetime(training_data.X_test.date.unique())

    lommersum_df = prepare_weather(lommersum_df, "lommersum_")
    koelnbonn_df = prepare_weather(koelnbonn_df, "koelnbonn_")

    ext_datapoints_df = lommersum_df.join(other=koelnbonn_df, on="date")
    ext_datapoints_df.drop(index=test_dates, inplace=True, errors="ignore")

    return ext_datapoints_df


WEATHER_DF = _load_weather_df()


class Bin:
    ALL = "ALL"
    DRY_SUNNY = "DRY_SUNNY"
    DRY_OVERCAST = "DRY_OVERCAST"
    SLIGHT_RAIN = "SLIGHT_RAIN"
    HEAVY_RAIN = "HEAVY_RAIN"


# All weather bins except for ALL
ALL_WEATHER_BINS = [
    Bin.DRY_SUNNY,
    Bin.DRY_OVERCAST,
    Bin.SLIGHT_RAIN,
    Bin.HEAVY_RAIN,
]


def get_weather_data_for_bin(month: int) -> pd.DataFrame:
    """Get all weather datapoints that match `bin` and are in `month`.

    The datapoints are selected by comparing "lommersum_precipitation_height" and
    "lommersum_sunshine_duration" with fixed cutoff values. The cutoff values were
    manually selected based on the attribute distribution.

    Args:
        month (int): month for which to request weather data
        bin (WeatherBin): profile for which to request weather data

    Returns:
        pd.DataFrame: all rows from WEATHER_DF from `month` with additional boolean
        columns DRY_SUNNY, DRY_OVERCAST, SLIGHT_RAIN, HEAVY_RAIN, which is True if the
        row belongs to that weather bin.
    """

    df = WEATHER_DF[WEATHER_DF.index.month == month].copy()
    df[Bin.SLIGHT_RAIN] = np.logical_and(
        df.lommersum_precipitation_height >= 0.2, df.lommersum_precipitation_height < 3
    )
    df[Bin.HEAVY_RAIN] = df.lommersum_precipitation_height >= 3
    df[Bin.DRY_SUNNY] = np.logical_and(
        df.lommersum_precipitation_height < 0.2, df.lommersum_sunshine_duration >= 4.5
    )
    df[Bin.DRY_OVERCAST] = np.logical_and(
        df.lommersum_precipitation_height < 0.2, df.lommersum_sunshine_duration < 4.5
    )

    return df


def _bin_for_weather_data(row: pd.Series) -> str:

    if row.lommersum_precipitation_height >= 0.2:
        if row.lommersum_precipitation_height >= 3:
            return Bin.HEAVY_RAIN
        else:
            return Bin.SLIGHT_RAIN
    else:
        if row.lommersum_sunshine_duration >= 4.5:
            return Bin.DRY_SUNNY
        else:
            return Bin.DRY_OVERCAST


def get_bin_for_weather_data(df: pd.DataFrame) -> pd.Series:
    """Calculate the weather bin for each weather datapoint (`df` may also contain 
    additional columns apart from weather).

    Args:
        df (pd.DataFrame): each row contains weather information like in `X_train`.

    Returns:
        pd.Series: a list of weather bins, one for each row in `df`. This will never be 
        ALL.
    """

    bins = df.apply(_bin_for_weather_data, axis="columns")

    return bins
