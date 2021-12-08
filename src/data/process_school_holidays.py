"""
Project: Phantasialand
State: 10/2021

Convert German school holiday information in a custom format (i.e. scraped from schulferien.org)
to a csv file. The resulting csv files has dates as rows and federal states as columns. If there is
a holiday at a given date in a given state, the cell contains the holiday name (e.g. Sommerferien).
Otherwise the cell is empty. The csv file contains only dates that are a holiday in at least one 
state. It is also possible to output the data in long form, i.e. with the columns date, federal
state and holiday type.
"""

import datetime
from typing import List, Optional

import pandas as pd
import click

from src.data.constants import STATE_FULL2ISO


def parse_relative_date(date_str: str, year: int) -> datetime.date:
    """parse a string of the form "28.02.".

    Args:
        date_str (str): date string, e.g. "28.02."
        year (int): corresponding year, e.g. 2019

    Returns:
        datetime.date: combined date object
    """

    part_list = date_str.strip().split(".")

    # splitting results in three parts with the third being empty
    if len(part_list) != 3 or part_list[2]:
        raise ValueError(f"illegal date format: {date_str}")

    return datetime.date(year=year, month=int(part_list[1]), day=int(part_list[0]))


def parse_date_string(date_str: str, year: int) -> List[str]:
    """parse date intervals into a list of ISO dates

    types of date intervals:
    - "-" (empty list)
    - "31.05.+11.06." (two single dates)
    - "13.05. - 17.05.+31.05." (interval and single date)
    - "04.10.+07.10. - 19.10." (single date and interval)
    - "04.10.+07.10. - 12.10.+01.11." (interval and two single dates)

    Parsing is done by the following steps:
    1. split the line at "+" and treat each part individually
    2a. If a part contains "-": Split, parse both parts with `parse_relative_date` and
        iteratively add all dates between the first and the last (inclusive) to the
        output list.
    2b. Else: Parse the date with `parse_relative_date` and add it to the output list as
        well

    Args:
        date_str (str): date string containing at most one interval and at most two
            single dates
        year (int): the year of the given dates

    Returns:
        List[str]: list of all dates that are described by `date_str` in ISO form.
    """

    date_str = date_str.strip()

    if date_str == "-":
        return []

    date_list = []

    date_parts = date_str.split("+")

    if len(date_parts) > 3:
        raise ValueError(f"illegal date string '{date_str}', too many '+'")

    for part in date_parts:

        if "-" in part:
            # this part is an interval

            interval_parts = part.split("-")

            if len(interval_parts) != 2:
                raise ValueError(f"illegal time interval '{part}' in line '{date_str}'")

            start_date = parse_relative_date(interval_parts[0], year)
            end_date = parse_relative_date(interval_parts[1], year)
            delta = datetime.timedelta(days=1)

            if end_date < start_date:
                # deal with holidays that contain a New Years Eve, e.g. 23.12.-6.1.
                end_date = datetime.date(
                    day=end_date.day, month=end_date.month, year=end_date.year + 1
                )

            while start_date <= end_date:

                date_list.append(start_date.isoformat())

                start_date += delta

        else:
            # this part is a single date

            date = parse_relative_date(part, year)

            date_list.append(date.isoformat())

    return date_list


def process_sections(line_list: List[str], holiday_names: List[str]) -> pd.DataFrame:
    """process the different sections of the input file, i.e. everything except for the header

    Args:
        line_list (List[str]): list of stripped lines, empty lines are removed
        holiday_names (List[str]): list of all holiday names

    Returns:
        pd.DataFrame: long-form dataframe containing date, federal state and holiday type as columns
    """

    date_state_holiday_list = []

    while line_list:

        # start new year
        year_line = line_list.pop(0)
        assert year_line.startswith("#"), year_line
        year = int(year_line[1:].strip())

        # iterate over states within that year
        while line_list and not line_list[0].startswith("#"):

            state = line_list.pop(0).strip()

            for holiday in holiday_names:
                curr_line = line_list.pop(0)

                assert curr_line[0].isdigit() or curr_line[0] == "-", curr_line

                date_list = parse_date_string(curr_line, year)

                for date in date_list:
                    date_state_holiday_list.append((date, state, holiday))

    raw_df = pd.DataFrame(date_state_holiday_list, columns=["date", "state", "type"])

    return raw_df


@click.command(help=__doc__)
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option(
    "--long-form",
    default=None,
    type=click.Path(),
    help="Where to store the data in long form "
    "(i.e. columns: date, federal state, holiday type). Optional.",
)
def process_file(input_path: str, output_path: str, long_form: Optional[str]):
    """Extract the relevant information from a custom school holiday file.

    Args:
        input_path (str): Path to input file
        output_path (str): Where to store the pivotized DataFrame
                (rows: dates, columns: federal states, cells: holiday type or empty)
        long_form (str): Where to store the long-form DataFrame
                (columns: dates, federal states, holiday type). Optional.

    Raises:
        ValueError: Raised if the input file is ill-formed
    """

    with open(input_path) as fp:
        line_list = fp.readlines()

    # strip and remove empty lines
    line_list = [stripped_line for line in line_list if (stripped_line := line.strip())]

    # we expect the file to start with "# Header" or something similar, followed by the
    # names of the 6 different holiday names
    holiday_names = line_list[1:7]

    # after the holiday names, the actual content of the file follows. It consists of
    # sections separated by "# <YEAR>"
    if not line_list[0].startswith("#") or not line_list[7].startswith("#"):
        raise ValueError(
            f"unknown file format, can only work with files containing "
            "{len(holiday_names)} different holidays"
        )

    line_list = line_list[7:]

    # process the sections containing the file content
    raw_df = process_sections(line_list, holiday_names)

    # use iso codes instead of full names
    raw_df.state.replace(STATE_FULL2ISO, inplace=True)
    raw_df.index.name = "id"

    if long_form:
        raw_df.to_csv(long_form)

    # pivotize the table. This ensures that each date appears at most once as row in the
    # output data and all information regarding this data is part of that row.
    pivot_df = raw_df.pivot(index="date", columns="state")
    pivot_df.columns = pivot_df.columns.droplevel(0)

    pivot_df.to_csv(output_path)


if __name__ == "__main__":

    process_file()
