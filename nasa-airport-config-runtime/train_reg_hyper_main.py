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
import threading

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
    .sample(2000, random_state=11)
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

    # make_all_predictions(airport_config_df_map, submission)

    airport_scores = []
    lookahead_scores = []
    max_airport_lookahead_score=[]
    for i in range(10):
        max_airport_lookahead_score.append([])
        for j in range(12):
            max_airport_lookahead_score[i].append(5)
    counter =0
    running_threads = []
    lock = threading.Lock()
    for hedge in  [.25,1,1.5]:
        for weights in [3, 6, 9]:
            def calculations(airport_config_df_map, submission_format, hedge, weights, training_labels, max_airport_lookahead_score, counter, lock):
                submission = submission_format.copy()
                submission["active"] = np.nan
                make_all_predictions_test(airport_config_df_map, submission, hedge = hedge, weight = weights)
                # submission.to_csv(prediction_path, date_format=DATETIME_FORMAT, index=False)
                scores = [
                (tup, log_loss(group["active"], submission.loc[group.index, "active"]))
                for tup, group in training_labels.groupby(["airport", "lookahead"])
                ]
                # scores: [(airport, lookahead), score]
                lock.acquire()
                for i in range(10):
                    for j in range(12):
                    # max_airport_lookahead_score[i,j] = [hedge, weights, score]
                        if not isinstance(max_airport_lookahead_score[i][j], list) or max_airport_lookahead_score[i][j][2] > scores[i*12+j][1]:
                            max_airport_lookahead_score[i][j] = [hedge, weights, scores[i*12+j][1]]
                
                lock.release()
                print(counter)
                print(max_airport_lookahead_score)
            counter += 1

            newThread = threading.Thread(target=calculations, args=(airport_config_df_map, submission_format, hedge, weights, training_labels, max_airport_lookahead_score, counter, lock,))
            print(counter)
            newThread.start()
            running_threads.append(newThread)
            if counter % 1 == 0:
                for t in running_threads:
                    print(counter)
                    t.join()
                running_threads = []
    
    for t in running_threads:
        print(counter)
        t.join()
    print(max_airport_lookahead_score)
    
    




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

if __name__ == "__main__":
    typer.run(main)

