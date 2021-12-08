"""
Project: Phantasialand
State: 11/2021

Perform E2E evaluations of models (either trained models or MeanEstimator) and store the
waiting time estimates. This data can later be used by the `fm_e2e_evaluate_xxx`
notebooks to visualize the model performance.
"""
from os import PathLike
from typing import Optional
from tqdm import tqdm
import pandas as pd
import click

from src.training.utils import load_data
from src.models.weather_bins import get_bin_for_weather_data
from src.models.mean_estimator import MeanEstimator
from src.models.model_estimator import ModelEstimator


@click.command(help=__doc__)
@click.argument("output_path", type=click.Path())
@click.option(
    "-m",
    "--model-uri",
    "model_uri",
    help="URI of the model to test. If not given, MeanEstimator is tested",
    default=None,
)
def main(output_path: PathLike, model_uri: Optional[str]):

    data = load_data()

    # The end user waiting time prediction works based on day and attraction only, so we
    # only need one row per day 
    days_df = data.X_test.drop(columns="half_hour_time").drop_duplicates()
    days_df["weather_bin"] = get_bin_for_weather_data(days_df)

    if model_uri:
        model = ModelEstimator(model_uri)
    else:
        model = MeanEstimator()

    dfs = []

    
    for _, row in tqdm(days_df.iterrows(), total=len(days_df)):

        # predict waiting times for every half hour of the current date and attraction
        time_by_weather_df, _ = model.predict(pd.to_datetime(row.date), row.attraction)

        dfs.append(
            pd.DataFrame(
                {
                    "date": row.date,
                    "attraction": row.attraction,
                    "half_hour_time": time_by_weather_df.index,
                    "y_pred": time_by_weather_df[row.weather_bin],
                }
            )
        )

    pred_df = pd.concat(dfs, ignore_index=True)

    true_df = data.X_test.copy()
    true_df["y_true"] = data.y_test

    pred_df.set_index(["attraction", "date", "half_hour_time"], inplace=True)
    true_df.set_index(["attraction", "date", "half_hour_time"], inplace=True)

    # merge predictions and actual values. The half_hour_time's present in both 
    # dataframes might slightly differ, because the prediction works with fixed opening 
    # hours, therefore an outer join is used.
    merge_df = pd.merge(
        pred_df, true_df, how="outer", left_index=True, right_index=True
    )

    merge_df.to_csv(output_path)


if __name__ == "__main__":
    main()
