"""
Project: Phantasialand
State: 11/2021

Take the processed waiting time and weather data and produce a train test split

This script joins the waiting time data with the weather data of both weather stations. 
The resulting datapoints are splitted into test and train set while ensuring that all 
datapoints from one day are part of the same set.

This script reads the waiting time and weather data from 
"data/interim/waiting_times_training.csv", 
"data/interim/weather_station01327_Lommersum.csv" and 
"data/interim/weather_station02667_Koeln-Bonn.csv" and writes the processed data to 
"data/processed/X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv".
"""

from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split
import click

from src.data.constants import DATA_PATH


def prepare_weather(weather_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """remove columns where all values are NaN and add a prefix to each column name

    Args:
        weather_df (pd.DataFrame): DataFrame to modify
        prefix (str): prefix string

    Returns:
        pd.DataFrame: modified DataFrame
    """
    return weather_df.dropna(axis="columns", how="all").add_prefix(prefix)


def train_test_split_date_based(
    df: pd.DataFrame, test_size: float
) -> Dict[str, pd.DataFrame]:
    """split waiting time data into train and test set but ensure that all datapoints of
    one day become part of the same dataset.

    Args:
        df (pd.DataFrame): DataFrame with waiting time information
        test_size (float): test set proportion

    Returns:
        Dict[str, pd.DataFrame]: dictionary mapping 'X_train', 'X_test', 'y_train',
        'y_test' to the respective DataFrames.
    """

    date_train, date_test = train_test_split(
        df.date.unique(),
        test_size=test_size,
        random_state=42,
        shuffle=True,
    )

    train_df = df.query("date in @date_train")
    test_df = df.query("date in @date_test")

    output_dfs = {
        "X_train": train_df.drop(columns=["waiting_time"]),
        "y_train": train_df.waiting_time,
        "X_test": test_df.drop(columns=["waiting_time"]),
        "y_test": test_df.waiting_time,
    }

    return output_dfs


@click.command(help=__doc__)
@click.argument("output_dir", type=click.Path())
def main(output_dir):
    """read and process waiting time and weather data. Afterwards, join the data,
    perform a train test split and save it as CSV files.
    """

    waiting_time_df = pd.read_csv(
        DATA_PATH / "interim/waiting_times_training.csv",
        index_col="id",
        parse_dates=["date"],
    )
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

    lommersum_df = prepare_weather(lommersum_df, "lommersum_")
    koelnbonn_df = prepare_weather(koelnbonn_df, "koelnbonn_")

    ext_datapoints_df = waiting_time_df.join(other=lommersum_df, on="date").join(
        other=koelnbonn_df, on="date"
    )

    split_dfs = train_test_split_date_based(ext_datapoints_df, 0.2)

    print(
        f"{len(split_dfs['X_train'])} train samples, {len(split_dfs['X_test'])} test samples"
    )
    print(
        f"proportion of test samples: {len(split_dfs['X_test'])/(len(split_dfs['X_test'])+len(split_dfs['X_train'])):.2%}"
    )

    ext_datapoints_df.to_csv(f"{output_dir}/all_datapoints.csv", index=False)

    for name, data in split_dfs.items():
        data.to_csv(f"{output_dir}/{name}.csv", index=False)


if __name__ == "__main__":
    main()
