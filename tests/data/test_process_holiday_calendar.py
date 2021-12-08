import unittest

from src.data.process_public_holidays import transform_dataframe
import pandas as pd
from datetime import date

class TestProcessPublicHolidays(unittest.TestCase):

    def test_exceptions_transform_dataframe(self):
        
        duplicate_day_df = pd.DataFrame([
            {
                "uid": "123", 
                "name": "Pfingsmontag", 
                "begin": date(2017, 6, 4),
                "location": "BB",
            },
            {
                "uid": "124", 
                "name": "Pfingstmontag Vol. 2", 
                "begin": date(2017, 6, 4),
                "location": "BB",
            }
        ])

        with self.assertRaises(ValueError):
            transform_dataframe(duplicate_day_df)

if __name__ == "__main__":

    unittest.main()