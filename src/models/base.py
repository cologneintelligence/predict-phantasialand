from abc import ABC, abstractmethod
import datetime

import pandas as pd

class WeatherBinEstimator(ABC):
    """Base class for all models that create predictions in the form we want to present
    to the end user.
    """

    @abstractmethod
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
        pass