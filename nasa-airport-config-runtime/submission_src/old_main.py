#FOR THE GLORY!!

"""Implementation of the Recency-weighted historical forecast solution to the Run-way Functions:
Predict Reconfigurations at US Airports challenge.
https://www.drivendata.co/blog/airport-configuration-benchmark/
"""

from datetime import datetime
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from src.utils import make_all_predictions, read_airport_configs

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

feature_directory = Path("/codeexecution/data")
prediction_path = Path("/codeexecution/prediction.csv")


def main(prediction_time: datetime):
    logger.info("Computing my predictions for {}", prediction_time)

    submission_format = pd.read_csv(
        feature_directory / "partial_submission_format.csv", parse_dates=["timestamp"]
    )

    airport_directories = sorted(path for path in feature_directory.glob("k*"))
    print(airport_directories)
#    airport_config_df_map = {}
#    for airport_directory in sorted(airport_directories):
#        airport_code, airport_config_df = read_airport_configs(airport_directory)
#        print(airport_code)
#        airport_config_df_map[airport_code] = airport_config_df
#    for airport_code, df in airport_config_df_map.items():
#        print(f"{airport_code}: {len(df):>8,}")
#    #    submission_format = pd.read_csv(
#    #        feature_directory / "partial_submission_format.csv", parse_dates=["timestamp"]
#    #    )
#    print(f"{len(submission_format):,} rows x {len(submission_format.columns)} columns")

#    submission = submission_format.copy()
#    submission["active"] = np.nan

#    make_all_predictions(airport_config_df_map, submission)

#    submission.to_csv(prediction_path, date_format=DATETIME_FORMAT, index=False)


if __name__ == "__main__":
    typer.run(main)