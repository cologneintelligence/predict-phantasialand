"""
Project: Phantasialand  
State: 11/2021

Create a pipeline to turn heterogenous training/test data into a feature matrix 
consisting only of floats.

Conversions performed:
- `attraction`: one-hot-encoding
- `time`: converted to floats (e.g. 08:30:00 -> 8.5)
- `date`: extract various date-based features (month, day, weekday, week of year, ...)
    and extract whether the day is a public and/or school holiday
"""
from typing import Iterable
import logging

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
import click
import joblib

from src.data.constants import (
    DATA_PATH,
    LOGGING_FORMAT_STR,
    STATE_FULL2ISO,
    WARTEZEITEN_APP_ATTRACTIONS,
)

# Some parameters describing how build_features currently works. These values are logged
# to mlflow to make it easier to see which training run used which featurization
FEATURIZATION_PARAMS = {
    "date_cols": "day_of_week,week_of_year",
    "weather_cols": "lommersum_precipitation_height,lommersum_sunshine_duration,lommersum_mean_temperature",
    "StandardScaler_with_mean": False
}

# This is a list of all weather columns in the input data. Names are commented out to
# indicate that they are (currently) not used and why.
SELECTED_WEATHER_COLUMNS = [
    # "lommersum_quality_other",         data quality attribute
    "lommersum_precipitation_height",
    # "lommersum_precipitation_form",    redundant to precipitation_height
    "lommersum_sunshine_duration",
    # "lommersum_snow_depth",            predictions are better without this feature
    # "lommersum_mean_vapor_pressure",   correlated with mean_temperature
    "lommersum_mean_temperature",
    # "lommersum_mean_relative_humidity",predictions are better without this feature
    # "lommersum_max_temperature_2m",    correlated with mean_temperature
    # "lommersum_min_temperature_2m",    correlated with mean_temperature
    # "lommersum_min_temperature_5cm",   correlated with mean_temperature
    # "koelnbonn_quality_wind",          quality attribute
    # "koelnbonn_max_wind_gust",         correlated with mean_wind_speed
    # "koelnbonn_mean_wind_speed",       predictions are better without this feature
    # "koelnbonn_quality_other", #       data quality attribute
    # "koelnbonn_precipitation_height",  also present at Lommersum
    # "koelnbonn_precipitation_form",    same
    # "koelnbonn_sunshine_duration",     same
    # "koelnbonn_snow_depth",            same
    # "koelnbonn_mean_cloud_cover",      predictions are better without this feature
    # "koelnbonn_mean_vapor_pressure",   also present at Lommersum
    # "koelnbonn_mean_pressure",         predictions are better without this feature
    # "koelnbonn_mean_temperature",      also present at Lommersum
    # "koelnbonn_mean_relative_humidity",same
    # "koelnbonn_max_temperature_2m",    same
    # "koelnbonn_min_temperature_2m",    same
    # "koelnbonn_min_temperature_5cm",   same
]


_PUBLIC_HOLIDAYS = pd.read_csv(
    DATA_PATH / "processed/public_holidays.csv", index_col="date", parse_dates=["date"]
)
_SCHOOL_HOLIDAYS = pd.read_csv(
    DATA_PATH / "processed/school_holidays.csv", index_col="date", parse_dates=["date"]
).fillna("")

# TODO move this to data preprocessing scripts
# We need to join public and school holidays with the input data anyways, so we can
# pre-execute this part of the joining
HOLIDAYS = pd.merge(
    _PUBLIC_HOLIDAYS,
    _SCHOOL_HOLIDAYS,
    how="outer",
    left_on="date",
    right_on="date",
    suffixes=("_public", "_school"),
    copy=False,
)

# We use three different categories when dealing with states: NRW, neighboring states
# and other states
NEIGHBOR_STATE_CODES = ["NI", "HE", "RP"]
OTHER_STATE_CODES = list(
    set(STATE_FULL2ISO.values()) - set(NEIGHBOR_STATE_CODES) - {"NW"}
)



def transform_time(times: Iterable[str]) -> pd.DataFrame:
    """convert a time in HH:MM:00 format to a float.

    Example:
        08:30 is converted into 8.5.

    Args:
        times (Iterable[str]): list of time strings in HH:MM:00 format.

    Returns:
        pd.DataFrame: one-column dataframe containing float representations.
    """

    output = []

    for time in times:

        hours, minutes, _ = time.split(":", maxsplit=2)

        output.append(float(hours) + float(minutes) / 60)

    return pd.DataFrame(output)


def transform_date(dates: Iterable[str]) -> pd.DataFrame:
    """extract date-based information.

    The following features are extracted: weekday, week number

    Args:
        date (Iterable[str]): dates in the YYY-MM-DD format.

    Returns:
        pd.DataFrame: DataFrame containing only integers, one row for for each date.
    """

    dates = pd.to_datetime(pd.Series(dates))

    datebased_df = pd.DataFrame(
        {
            "day_of_week": dates.dt.day_of_week,
            "week_of_year": dates.dt.isocalendar().week.astype("int64"),
        }
    )

    return datebased_df


# Python dicts are insertion-ordered as of 3.7, so this always reflects the order of 
# entries in datebased_df.
DATE_COLUMNS = [
    "day_of_week",
    "week_of_year",
]


def transform_holidays(dates: Iterable[str]) -> pd.DataFrame:
    """extract holiday-based information.

    The following features are extracted: public_holiday_NW, public_holiday_neighbors,
    public_holiday_others, school_holiday_NW, school_holiday_neighbors,
    school_holiday_others.

    Neighbors are Niedersachsen, Hessen and Rheinland-Pfalz.

    Args:
        date (Iterable[str]): dates in the YYY-MM-DD format.

    Returns:
        pd.DataFrame: DataFrame containing only 1.0 or 0.0, one row for for each date.
    """

    # Left join of `dates` to get the holiday information for each date (and Nan/"" if
    # a date is neither of both)
    dates = pd.to_datetime(pd.Series(dates))
    dates.name = "date"

    df = pd.merge(
        dates,
        HOLIDAYS,
        how="left",
        left_on="date",
        right_on="date",
        suffixes=("_dates", None),
    )
    df.fillna(False, inplace=True)

    # STATE_school contains strings, but as we want to interpret empty strings as False
    # and non-empty strings as True, we can mostly treat them like the boolean values in
    # STATE_public.

    result_df = pd.DataFrame(
        {
            "public_holiday_NW": df.NW_public.astype("float"),
            "public_holiday_neighbors": (
                df[[f"{code}_public" for code in NEIGHBOR_STATE_CODES]]
                .any(axis="columns")
                .astype("float")
            ),
            "public_holiday_others": (
                df[[f"{code}_public" for code in OTHER_STATE_CODES]]
                .any(axis="columns")
                .astype("float")
            ),
            # we need to convert twice here, as NW_y is a string series
            "school_holiday_NW": df.NW_school.astype("bool").astype("float"),
            "school_holiday_neighbors": (
                df[[f"{code}_school" for code in NEIGHBOR_STATE_CODES]]
                .any(axis="columns")
                .astype("float")
            ),
            "school_holiday_others": (
                df[[f"{code}_school" for code in OTHER_STATE_CODES]]
                .any(axis="columns")
                .astype("float")
            ),
        }
    )

    return result_df


HOLIDAY_COLUMNS = [
    "public_holiday_NW",
    "public_holiday_neighbors",
    "public_holiday_others",
    "school_holiday_NW",
    "school_holiday_neighbors",
    "school_holiday_others",
]


def build_pipeline(model: BaseEstimator = None):
    """construct a Pipeline object containing all preprocessing steps and optionally a
    model as final step.

    Args:
        model (BaseEstimator): final step in the pipeline, i.e. the estimator. Optional.

    Returns:
        Pipeline: sklearn pipeline
    """

    date_transformer = FunctionTransformer(transform_date)
    holiday_transformer = FunctionTransformer(transform_holidays)
    time_transformer = FunctionTransformer(transform_time)
    attraction_transformer = OneHotEncoder(
        drop=None, categories=[list(WARTEZEITEN_APP_ATTRACTIONS.keys())]
    ) 
    weather_transformer = SimpleImputer(missing_values=np.nan, strategy="mean")

    date_holiday_transformer = FeatureUnion(
        [
            ("date_transformer", date_transformer),
            ("holiday_transformer", holiday_transformer),
        ]
    )

    column_transformer = ColumnTransformer(
        [
            ("weather", weather_transformer, SELECTED_WEATHER_COLUMNS),
            ("date", date_holiday_transformer, "date"),
            ("time", time_transformer, "half_hour_time"),
            ("attraction", attraction_transformer, ["attraction"]),
        ]
    )

    components = [
        ("column_transformer", column_transformer),
        ("standard_scaler", StandardScaler()),
    ]

    if model:
        components.append(("model", model))

    preprocessing_pipeline = Pipeline(components)

    return preprocessing_pipeline


FEATURE_MATRIX_COLUMNS = (
    SELECTED_WEATHER_COLUMNS
    + DATE_COLUMNS
    + HOLIDAY_COLUMNS
    + ["time"]
    + list(WARTEZEITEN_APP_ATTRACTIONS.keys())
)


@click.command(help=__doc__)
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_file_path", type=click.Path())
@click.argument("model_path", type=click.Path())
def featurize_test_train(input_dir, output_file_path, model_path):
    """apply featurization pipeline on X_train and X_test, saving the feature matrices
    and the fitted pipeline.

    This is only for testing/debugging purposes. During model training the whole 
    pipeline is fitted so there is no need for intermediate outputs.

    Args:
        output_file_path (str): where to store the feature matrices (via np.savez)
        model_path (str): where to store the fitted pipeline (via joblib)
    """
    logging.basicConfig(format=LOGGING_FORMAT_STR, level=logging.DEBUG)

    logging.info("Building pipeline...")
    pipeline = build_pipeline()

    logging.info("Reading X_train...")
    X_train = pd.read_csv(f"{input_dir}/X_train.csv")

    logging.info("Fitting and transforming X_train...")
    X_train_p = pipeline.fit_transform(X_train)

    logging.info("Reading X_test")
    X_test = pd.read_csv(f"{input_dir}/X_test.csv")

    logging.info("Transforming X_test")
    X_test_p = pipeline.transform(X_test)

    logging.info("Saving data and model")
    np.savez_compressed(output_file_path, X_train_p=X_train_p, X_test_p=X_test_p)

    joblib.dump(pipeline, model_path)


if __name__ == "__main__":

    featurize_test_train()


