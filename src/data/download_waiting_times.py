"""
Project: Phantasialand
State: 10/2021

Download waiting time information from wartezeiten.app using the undocumented internal
API.

Waitings times are downloaded by querying `www.wartezeiten.app/charts/linechart.php`
which returns a list of datapoints in the form of ({"datum":"2021-08-01
09:00:12","wartezeit":"-3", "status":"closed"}) for a given amusement park, attraction,
month and year. It takes internal IDs as parameters which must be listed in ATTRACTIONS,
MONTHS and YEARS. This script then queries all possible combinations of ATTRACTIONS,
MONTHS and YEARS. Querying non-existing documents is no problem.

The queried data ist stored in one csv-file with the following columns (all entries are
strings): attraction, month, year, datum, wartezeit, status 

The output should be further processed with `process_wartezeiten_app.py`
"""

from typing import Optional, List, Dict
import itertools
import logging

import requests
from tqdm import tqdm
import pandas as pd
import click

from src.data.constants import (
    WARTEZEITEN_APP_ATTRACTIONS,
    WARTEZEITEN_APP_MONTHS,
    WARTEZEITEN_APP_YEARS,
    LOGGING_FORMAT_STR,
)


def query_waiting_times(
    attraction: str, month: str, year: str
) -> List[Dict[str, str]]:
    """query the waiting times for a single attraction and a given month and year.

    Upon sucess it returns a list data points. If there is not information
    for the given date, None is returned. If the server returns something
    weird an exception is raised.

    Args:
    attraction (str): internal attraction id, e.g. "636a733d"
    month (str): internal month id, e.g. "63673d3d"
    year (str): internal year id, e.g. "6354303965413d3d"

    Returns:
    list[dict[str, str]]: a list of data points in the form of
        {"datum":"2021-08-01 09:00:12","wartezeit":"-3","status":"closed"} or
        None if no information is returned, e.g. when querying a month for which
        no data is present.
    """
    r = requests.get(
        "https://www.wartezeiten.app/charts/linechart.php",
        params={
            "park": "4d3256754a354f354d503567464d316a7a773d3d",
            "code": attraction,
            "monat": month,
            "jahr": year,
        },
    )

    # when querying a month/year combination that does not exists, wartezeiten still
    # returns status code 200 and 'null' as content. Even when sending completely wrong
    # parameters it still returns 200 and 'null' or '' depending on the parameter.

    r.raise_for_status()

    if r.text:
        return r.json()
    else:
        return []



def download_all_waiting_times() -> List[dict]:
    """download the waitings times for all attractions, months and years.

    Returns:
    List[dict]: one entry for each queried document
    """

    requested_data = []
    errors = []

    total_iterations = (
        len(WARTEZEITEN_APP_ATTRACTIONS)
        * len(WARTEZEITEN_APP_MONTHS)
        * len(WARTEZEITEN_APP_YEARS)
    )

    for ((attr_name, attr_id), (month_name, month_id), (year_name, year_id)) in tqdm(
        itertools.product(
            WARTEZEITEN_APP_ATTRACTIONS.items(),
            WARTEZEITEN_APP_MONTHS.items(),
            WARTEZEITEN_APP_YEARS.items(),
        ),
        total=total_iterations,
    ):
        try:
            datapoints = query_waiting_times(attr_id, month_id, year_id)
            if datapoints:
                data = {
                    "attraction": attr_name,
                    "month": month_name,
                    "year": year_name,
                    "data": datapoints,
                }
                requested_data.append(data)
        except Exception as e:
            logging.error(f"Download exception (This is not necessarily a problem): "
                            f"{attr_name=}, {month_name=}, {year_name=}, {e=}")

    return requested_data


def convert_to_dataframe(raw_data: List[dict]) -> pd.DataFrame:
    """convert the list returned by `download_all_waiting_times` to a flat DataFrame.

    Args:
    raw_data (List[dict]): list of queried documents

    Returns:
    pd.DataFrame: DataFrame containing one row per datapoints
    """

    flat_datapoints = []
    for document in tqdm(raw_data):
        for entry in document["data"]:
            flat_datapoints.append(
                {
                    "attraction": document["attraction"],
                    "month": document["month"],
                    "year": document["year"],
                    **entry,
                }
            )

    df = pd.DataFrame(flat_datapoints)
    df.index.name = "id"

    return df


@click.command(help=__doc__)
@click.argument("output_path", type=click.Path())
def main(output_path):
    logging.basicConfig(format=LOGGING_FORMAT_STR)

    logging.info("Download all combinations of attractions, month and year...")
    raw_data = download_all_waiting_times()

    logging.info("Convert to DataFrame...")
    df = convert_to_dataframe(raw_data)

    logging.info(f"Store data at {output_path} (csv-file)...")
    df.to_csv(output_path)


if __name__ == "__main__":

    main()
