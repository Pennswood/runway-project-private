
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

from src.utils import make_all_predictions, read_airport_configs



def compute_airport_top_k_accuracy(
    airport_predictions: pd.DataFrame,
    config_timecourse: pd.Series,
    k: int,
    only_change: bool = False,
) -> float:
    """Computes top k accuracy for a single airport"""
    airport = airport_predictions.iloc[0].airport
    airport_configs = airport_predictions.config.unique()
    top_k_hits = []
    for (timestamp, lookahead), group in airport_predictions.groupby(
        ["timestamp", "lookahead"]
    ):
        lookahead_config = config_timecourse.loc[
            timestamp + pd.Timedelta(minutes=lookahead)
        ]

        if lookahead_config not in airport_configs:
            lookahead_config = f"{airport}:other"
        
        if only_change:
            # only include samples where a configuration change occurred
            current_config = config_timecourse.loc[timestamp]
            include = current_config != lookahead_config
        else:
            include = True
        if include:
            top_k_predictions = group.sort_values(
                "active", ascending=False
            ).config.iloc[:k]
            top_k_hits.append(
                {
                    "lookahead": lookahead,
                    "hit": lookahead_config in top_k_predictions.values,
                }
            )

    return pd.DataFrame(top_k_hits).groupby("lookahead").hit.mean()

def compute_top_k_accuracy(airport_config_df_map: Dict[str, pd.DataFrame], ks: Sequence[int], predictions: pd.DataFrame, only_change: bool):
    """Iterates over airports to compute overall top k accuracy"""
    top_ks = {}
    for k in ks:
        airport_top_ks = []
        for airport in predictions.airport.unique():
            config_timecourse = (
                airport_config_df_map[airport]
                .set_index("timestamp")
                .airport_config.resample("30min")
                .ffill()
                .dropna()
            )
            config_timecourse = f"{airport}:" + config_timecourse
            airport_predictions = predictions.loc[predictions.airport == airport]
            airport_top_k = compute_airport_top_k_accuracy(
                airport_predictions, config_timecourse, k=k, only_change=only_change
            )
            airport_top_k.rename(airport, inplace=True)
            airport_top_ks.append(airport_top_k)
        top_ks[k] = pd.concat(airport_top_ks, axis=1).values.mean()

    return pd.Series(
        top_ks.values(), name="accuracy", index=pd.Index(top_ks.keys(), name="k")
    )



feature_directory = Path("./data")
prediction_path = Path("./test_prediction.csv")

submission = pd.read_csv(prediction_path, parse_dates=["timestamp"])
submission = (
    submission.set_index(["airport", "timestamp", "lookahead"])
    .reset_index()
    .sort_values(["airport", "timestamp", "lookahead", "config"])
)
airport_directories = sorted(path for path in feature_directory.glob("k*"))

airport_config_df_map = {}
for airport_directory in sorted(airport_directories):
    airport_code, airport_config_df = read_airport_configs(airport_directory)
    print(airport_code)
    airport_config_df_map[airport_code] = airport_config_df

top_k_all = compute_top_k_accuracy(airport_config_df_map, (1,2,3), submission, only_change=False)
top_k_only_change = compute_top_k_accuracy(airport_config_df_map, (1, 2,3), submission, only_change=True)
print(top_k_all)
print(top_k_only_change)


open_train_labels = pd.read_csv(
    feature_directory / "open_train_labels.csv.bz2", parse_dates=["timestamp"]
)
training_samples = (
open_train_labels.groupby(["airport", "lookahead"])
.sample(100, random_state=12)
.set_index(["airport", "timestamp", "lookahead"])
.sort_index()
)
training_labels = (
    open_train_labels.set_index(["airport", "timestamp", "lookahead"])
    .loc[training_samples.index]  # select the sampled timestamp/lookaheads
    .reset_index()
    .sort_values(["airport", "timestamp", "lookahead", "config"])
)

scores = [
(airport, log_loss(group["active"], submission.loc[group.index, "active"]))
for airport, group in training_labels.groupby(["airport", "lookahead"])
]
print("Scores: "+str(scores))
(airport, log_loss(group["active"], submission.loc[group.index, "active"]))
for airport, group in training_labels.groupby("lookahead")
]
print("Scores: "+str(scores))