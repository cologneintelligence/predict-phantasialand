"""
Project: Phantasialand
State: 11/2021

Convert an ical file containing public holiday information into a flat table and store it as csv.

Important: This script assumes that all holidays are only one day and that there are no two 
holidays on the same day. This conditions are checked and exceptions are raised if necessary.

The output file contains the following columns:
- date (str): date of the holiday (YYYY-MM-DD). This must be unique
- name (str): German name of the holiday
- is_public_holiday (bool): if the day is a real public holiday (the data source contains a few 
    days that are off school but no public holidays).
- <iso state code> (bool): for each of the 16 German states this indicates whether the day is a 
    holiday in this state 
"""

import datetime
from pathlib import Path

from icalendar import Calendar, Event
import recurring_ical_events
import pandas as pd
import click

from src.data.constants import STATE_FULL2ISO


def _event2dict(event: Event) -> dict:
    """extract name, begin, uid and location from an icalendar event.

    Args:
        event (Event): event to extract information from

    Returns:
        dict: dictionary containing name, begin, uid and location.
    """

    return {
        "name": str(event["summary"]),
        "begin": event["dtstart"].dt,
        "uid": str(event["uid"]),
        "location": str(event["location"]),
    }


def ical_to_dataframe(cal: Calendar) -> pd.DataFrame:
    """extract the relevant attributes from each event and store them in a dataframe.

    Args:
        cal (Calendar): ical calendar of holidays

    Returns:
        pd.DataFrame: relevant attributes of each calendar entry
    """

    dates = [e["dtstart"].dt for e in cal.walk() if "dtstart" in e]
    start_date = min(dates)
    end_date = max(dates)

    events = recurring_ical_events.of(cal).between(start_date, end_date)

    row_list = [_event2dict(e) for e in events]

    df = pd.DataFrame(data=row_list)
    df.set_index("uid", inplace=True)

    return df


def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """transform the raw calendar data in the structure described in this modules docstring

    Args:
        df (pd.DataFrame): dataframe containing the raw event data

    Raises:
        ValueError: raised if the dates are not unique all-day single-day events

    Returns:
        pd.DataFrame: table containing the relevant information for each holiday
    """

    df["is_public_holiday"] = df.name.apply(lambda name: "§" in name)
    df.name = df.name.str.replace(" (§)", "", regex=False)

    df.location.replace(
        {"Alle Bundesländer": ",".join(STATE_FULL2ISO.values())}, inplace=True
    )
    for state in STATE_FULL2ISO.values():
        df[state] = df.location.apply(lambda location_str: state in location_str)

    df["begin_date"] = df.begin.apply(datetime.date.isoformat)

    if not df.begin_date.is_unique:
        raise ValueError(
            "Cannot have two holidays on the same day as we want to use the day as"
            " primary key!"
        )

    df.drop(columns=["location", "begin"], inplace=True)

    df.set_index("begin_date", inplace=True)
    df.index.name = "date"

    return df


@click.command(help=__doc__)
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def main(input_path: Path, output_path: Path):

    with open(input_path) as fp:
        cal = Calendar.from_ical(fp.read())

    df = ical_to_dataframe(cal)
    df = transform_dataframe(df)

    df.to_csv(output_path)


if __name__ == "__main__":

    main()
