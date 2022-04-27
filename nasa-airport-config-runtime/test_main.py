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
import copy
from sklearn.metrics import log_loss
from typing import Sequence, Tuple, Dict
import matplotlib as mpl

from src.utils import make_all_predictions, read_airport_configs, make_all_predictions_test

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


#TODO Change HERE
feature_directory = Path("./data")
prediction_path = Path("./test_prediction.csv")

def main(prediction_time: datetime):
    logger.info("Computing my predictions for {}", prediction_time)

    open_train_labels = pd.read_csv(
        feature_directory / "open_train_labels.csv.bz2", parse_dates=["timestamp"]
    )
    training_samples = (
    open_train_labels.groupby(["airport", "lookahead"])
    .sample(1000, random_state=314)
    .set_index(["airport", "timestamp", "lookahead"])
    .sort_index()
    )
    training_labels = (
        open_train_labels.set_index(["airport", "timestamp", "lookahead"])
        .loc[training_samples.index]  # select the sampled timestamp/lookaheads
        .reset_index()
        .sort_values(["airport", "timestamp", "lookahead", "config"])
    )
    submission_format = training_labels.copy().assign(active=np.nan)

    airport_directories = sorted(path for path in feature_directory.glob("k*"))

    airport_config_df_map = {}
    for airport_directory in sorted(airport_directories):
        airport_code, airport_config_df = read_airport_configs(airport_directory)
        print(airport_code)
        airport_config_df_map[airport_code] = airport_config_df
    for airport_code, df in airport_config_df_map.items():
        print(f"{airport_code}: {len(df):>8,}")
    #    submission_format = pd.read_csv(
    #        feature_directory / "partial_submission_format.csv", parse_dates=["timestamp"]
    #    )
    print(f"{len(submission_format):,} rows x {len(submission_format.columns)} columns")

    submission = submission_format.copy()
    submission["active"] = np.nan

    make_all_predictions(airport_config_df_map, submission)
    submission.to_csv(prediction_path, date_format=DATETIME_FORMAT, index=False)

    scores = [
    (airport, log_loss(group["active"], submission.loc[group.index, "active"]))
    for airport, group in training_labels.groupby(["airport", "lookahead"])
    ]
    print("Scores: "+str(scores))
    scores = [
    (log_loss(group["active"], submission.loc[group.index, "active"]))
    for airport, group in training_labels.groupby(["airport"])
    ]
    print(np.mean(scores))




if __name__ == "__main__":
    typer.run(main)

