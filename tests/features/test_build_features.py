import unittest

from src.features import build_features
import pandas as pd
import numpy as np


class TestProcessPublicHolidays(unittest.TestCase):
    def test_transform_time(self):

        time2feature = {
            "08:00:00": 8.0,
            "08:30:00": 8.5,
            "10:00:00": 10.0,
        }

        expected_df = pd.DataFrame(time2feature.values())

        actual_df = build_features.transform_time(time2feature.keys())

        self.assertTrue(
            np.allclose(expected_df, actual_df),
            f"{expected_df=}, \n{actual_df=}, "
            f"\n{expected_df.dtypes=}, {actual_df.dtypes=}",
        )

    def test_transform_date(self):

        dates = [
            "2021-10-26",
            "2021-01-01",
            "2021-05-01",
        ]

        expected_df = pd.DataFrame(
            {
                "day_of_week": [1, 4, 5],
                "day_of_month": [26, 1, 1],
                "day_of_year": [299, 1, 121],
                "week_of_year": [43, 53, 17],
                "month_of_year": [10, 1, 5],
            }
        )

        actual_df = build_features.transform_date(dates)

        self.assertTrue(
            expected_df.equals(actual_df),
            f"{expected_df=}\n {actual_df=}"
            f"{expected_df.dtypes=}, {actual_df.dtypes=}",
        )

    def test_transform_holiday(self):

        columns = [
            "public_holiday_NW",
            "public_holiday_neighbors",
            "public_holiday_others",
            "school_holiday_NW",
            "school_holiday_neighbors",
            "school_holiday_others",
        ]

        dates2features = {
            # order of values in the tuple is the same as in `columns`
            "2020-05-01": (1.0, 1.0, 1.0, 0.0, 0.0, 0.0),  # Tag der Arbeit
            # Augsburger Friedensfest & Sommerferien
            "2020-08-08": (0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
            "2020-04-13": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),  # Ostermontag
            "2020-09-11": (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),  # Sommerferien BW
            "2024-07-06": (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),  # Sommerferien NI,HB,SN,ST,TH
            # transform_holidays must be able to work with non-unique dates
            "2024-07-06": (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),  # Sommerferien NI,HB,SN,ST,TH
            "2024-07-06": (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),  # Sommerferien NI,HB,SN,ST,TH
        }

        actual_df = build_features.transform_holidays(dates2features.keys())

        expected_df = pd.DataFrame(dates2features.values(), columns=columns)

        self.assertTrue(
            expected_df.equals(actual_df),
            f"{expected_df=}\n {actual_df=}"
            f"{expected_df.dtypes=}, {actual_df.dtypes=}",
        )


if __name__ == "__main__":

    unittest.main()
