"""
Project: Phantasialand  
State: 10/2021

Jointly process current and historical daily weather data from the Deutschen Wetter
Dienst for a single weather station. This script can directly work with the zip files
provided by the DWD. The current daily weather data zip files is called
`tageswerte_KL_<STATION_ID>_akt.zip` and the historical zip file is called
`tageswerte_KL_<STATION_ID>_<START_DATE>_<END_DATE>_hist.zip`.  

The resulting CSV file contains one row for each date between 2019-01-01 and the newest
date in the data. It contains all columns from the original data (except for station
id), but with a more readable name.
"""

from zipfile import ZipFile
from typing import Tuple, Union, IO
from os import PathLike

import pandas as pd
import numpy as np
import click

from src.data.constants import DWD_COLUMN_NAMES2DESCRIPTION


def extract_dwd_archive(zip_file: Union[str, PathLike, IO]) -> pd.DataFrame:
    """Extract the weather data from one DWD OpenData zip file.

    This function extracts the actual weather data from the zip file and transforms it
    to a DataFrame. The zip archive is expected to contain exactly one file starting
    with "produkt_klima", which is the CSV file to be parsed.

    Args:
        zip_file (str | PathLike | IO): zip file to process.

    Raises:
        ValueError: the archive contains multiple files starting with "produkt_klima"

    Returns: pd.DataFrame: weather data as DataFrame
    """

    with ZipFile(zip_file, mode="r") as archive:

        data_file_names = [
            name for name in archive.namelist() if name.startswith("produkt_klima")
        ]

        if len(data_file_names) > 1:
            raise ValueError(
                f"multiple data files in one zip archive, cannot decide."
                f" {zip_file=}, {data_file_names=}"
            )

        with archive.open(data_file_names[0]) as fp:
            df = pd.read_csv(fp, sep=";", parse_dates=["MESS_DATUM"])

    return df


def clean_dwd_data(df: pd.DataFrame) -> pd.DataFrame:
    """clean the extracted weather data.

    Fix column names, remove end-of-record-marker and use np.nan instead of -999.
    Modifies the DataFrame in place.

    Args:
        df (pd.DataFrame): extracted DWD weather data to clean.

    Returns:
        pd.DataFrame: cleaned DWD weather data.
    """

    df.replace(-999, np.nan, inplace=True)
    df.drop(columns=["eor"], inplace=True)  # drop end of record marker
    # remove whitespace in column headers
    df.columns = [col.strip() for col in df.columns]
    # make column names human-readable
    df.rename(columns=DWD_COLUMN_NAMES2DESCRIPTION, inplace=True)

    return df


def merge_historical_current(
    df_current: pd.DataFrame, df_historical: pd.DataFrame
) -> Tuple[pd.DataFrame, int]:
    """Merge current and historical data ranging from 2019-01-01 to the newest date.

    This function als sets the data as index (after making sure all dates are unique)
    and drops the weather station id after checking that all datapoints stem from the
    same weather station.

    Args:
        df_current (pd.DataFrame): current daily weather data processed via
            `process_dwd_archive` and `clean_dwd_data`.
        df_historical (pd.DataFrame): historical daily weather data processed via
            `process_dwd_archive` and `clean_dwd_data`.

    Raises:
        ValueError: either `df_current` or `df_historical` contain multiple entries for
            the same date
        ValueError: the merged entries do not all stem from the same weather station

    Returns:
        pd.DataFrame: merged dataframe
        int: id of the weather stations from which all datapoints originate
    """

    # there is a certain overlap between current and historical, therefore we only use
    # historical data that is not present in the current dataset. We also drop all
    # datapoints older than 2019 because we do not need weather data from 1937.
    df_historical_relevant = df_historical[
        (df_historical.date >= pd.to_datetime("2019-01-01"))
        & (df_historical.date < df_current.date.min())
    ]

    df_merge = df_current.append(df_historical_relevant)

    if not df_merge.date.is_unique:
        value_counts = df_merge.date.value_counts()
        raise ValueError(
            f"daily weather data contains duplicates: {value_counts[value_counts > 1]}"
        )

    df_merge.set_index("date", inplace=True)

    unique_station_ids = df_merge.station_id.unique()

    if len(unique_station_ids) > 1:
        raise ValueError(
            f"Cannot merge data from multiple weather stations. "
            f"station ids {unique_station_ids}"
        )

    (station_id,) = unique_station_ids

    df_merge.drop(columns=["station_id"], inplace=True)
    df_merge.sort_index(ascending=False, inplace=True)

    return df_merge, station_id


@click.command(help=__doc__)
@click.argument("current_path", type=click.Path(exists=True))
@click.argument("historical_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def main(current_path, historical_path, output_path):

    current_df = clean_dwd_data(extract_dwd_archive(current_path))
    historical_df = clean_dwd_data(extract_dwd_archive(historical_path))

    df, station_id = merge_historical_current(current_df, historical_df)

    print("Station ID: ", station_id)

    df.to_csv(output_path)


if __name__ == "__main__":

    main()
