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
    .sample(750, random_state=17)
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
    counter =1
    running_threads = []
    lock = threading.Lock()
    for hedge in  [.0625,.125,.25,.5,.75,1,1.25,1.5,2,2.5]:
        for weights in [2,2.5,3,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,20,25,30]:
            def calculations(airport_config_df_map, submission, hedge, weights, training_labels, max_airport_lookahead_score, counter, lock):
                
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
                    # max_airport_lookahead_score[i,j] = [hedge, weights, accuracy, score]
                        if not isinstance(max_airport_lookahead_score[i][j], list) or max_airport_lookahead_score[i][j][2] > scores[i*12+j][1]:
                            max_airport_lookahead_score[i][j] = [hedge, weights, scores[i*12+j][1]]
                
                lock.release()
                print(counter)
                print(max_airport_lookahead_score)
            counter += 1
            newThread = threading.Thread(target=calculations, args=(airport_config_df_map, submission, hedge, weights, training_labels, max_airport_lookahead_score, counter, lock,))
            print(counter)
            newThread.start()
            running_threads.append(newThread)
    
    for t in running_threads:
        print(counter)
        t.join()
    print(max_airport_lookahead_score)
    
    for i in range(10):
        for j in range(12):
            max_airport_lookahead_score[i][j] = max_airport_lookahead_score[i][j][:3]
    




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


[[[1, 10, 1, 0.01957239342740286], [1, 10, 1, 0.029699197434304665], [1, 10, 1, 0.037377139087743996], [1, 10, 1, 0.04511073600853215], [1, 10, 1, 0.08077450260331363], [1, 10, 1, 0.07037955646062383], [1, 10, 1, 0.08261363734087121], [1, 10, 1, 0.09304888525150176], [1, 10, 1, 0.07893114230726365], [1, 10, 1, 0.09668466605828294], [1, 10, 1, 0.12434125912381272], [1, 10, 1, 0.14107461135031316]], [[1, 10, 1, 0.05480592672396588], [1, 10, 1, 0.0857283780601349], [1, 10, 1, 0.09523364735686152], [1, 10, 1, 0.09320202804947343], [1, 10, 1, 0.15294629595101597], [1, 10, 1, 0.191088737304411], [1, 10, 1, 0.16584224568903205], [1, 10, 1, 0.21218397662671293], [1, 10, 1, 0.2045642169030637], [1, 10, 1, 0.2299110198893625], [1, 10, 1, 0.30419396124436343], [1, 10, 1, 0.22913868938753704]], [[1, 10, 1, 0.026536449220641575], [1, 10, 1, 0.0788229196904625], [1, 10, 1, 0.07328193035919912], [1, 10, 1, 0.08109247910797313], [1, 10, 1, 0.0964455976539969], [1, 10, 1, 0.12018290848382923], [1, 10, 1, 0.12161547825365603], [1, 10, 1, 0.12178926304193041], [1, 10, 1, 0.13210550417472303], [1, 10, 1, 0.1443776696031852], [1, 10, 1, 0.14243858387833597], [1, 10, 1, 0.14436872246067328]], [[1, 10, 1, 0.019262444217022934], [1, 10, 1, 0.04398714646941975], [1, 10, 1, 0.057039419921722435], [1, 10, 1, 0.06886595426379682], [1, 10, 1, 0.07987120195229382], [1, 10, 1, 0.08999427045579096], [1, 10, 1, 0.08456257708443454], [1, 10, 1, 0.09939003085758029], [1, 10, 1, 0.1223247224800517], [1, 10, 1, 0.15300966512010408], [1, 10, 1, 0.1667482190468372], [1, 10, 1, 0.1483039837063016]], [[1, 10, 1, 0.028834102746266476], [1, 10, 1, 0.059810951478396206], [1, 10, 1, 0.0665863178564697], [1, 10, 1, 0.08189955535647164], [1, 10, 1, 0.07521311338982413], [1, 10, 1, 0.10099474011084913], [1, 10, 1, 0.11975713849005591], [1, 10, 1, 0.14013686027799865], [1, 10, 1, 0.15492939554201338], [1, 10, 1, 0.1531066363145676], [1, 10, 1, 0.15435954858509976], [1, 10, 1, 0.1895915279218965]], [[1, 10, 1, 0.03263742620318307], [1, 10, 1, 0.08867255332679398], [1, 10, 1, 0.08488725663400411], [1, 10, 1, 0.12405602384134905], [1, 10, 1, 0.13730086495476698], [1, 10, 1, 0.13733923810866008], [1, 10, 1, 0.15264931290547473], [1, 10, 1, 0.18462190455815028], [1, 10, 1, 0.1633115213763429], [1, 10, 1, 0.17816897774704313], [1, 10, 1, 0.18807376524339292], [1, 10, 1, 0.19952688318012277]], [[1, 10, 1, 0.023361575898340915], [1, 10, 1, 0.049812118017584076], [1, 10, 1, 0.062091212135785556], [1, 10, 1, 0.0731252826475704], [1, 10, 1, 0.07425049466062504], [1, 10, 1, 0.10682917222804743], [1, 10, 1, 0.08703319388622559], [1, 10, 1, 0.11525486597399702], [1, 10, 1, 0.1271440359382743], [1, 10, 1, 0.13468272702800646], [1, 10, 1, 0.11714304069154625], [1, 10, 1, 0.15174769821205358]], [[1, 10, 1, 0.03970668713361964], [1, 10, 1, 0.08113526984843376], [1, 10, 1, 0.0771818426205294], [1, 10, 1, 0.103294152574109], [1, 10, 1, 0.11482745715512213], [1, 10, 1, 0.12842037204187726], [1, 10, 1, 0.13261605086884173], [1, 10, 1, 0.12580043605390523], [1, 10, 1, 0.12774214213249493], [1, 10, 1, 0.13648826472822545], [1, 10, 1, 0.15011590846494205], [1, 10, 1, 0.15327724417272887]], [[1, 10, 1, 0.02700682092314686], [1, 10, 1, 0.06075297587325542], [1, 10, 1, 0.04751000282187495], [1, 10, 1, 0.07000453302106535], [1, 10, 1, 0.09245630820834269], [1, 10, 1, 0.1170783862702247], [1, 10, 1, 0.12216491448225339], [1, 10, 1, 0.11686083726566927], [1, 10, 1, 0.1407276359314766], [1, 10, 1, 0.14067074464188511], [1, 10, 1, 0.17447007289268598], [1, 10, 1, 0.18018454101053985]], [[1, 10, 1, 0.036400598709406574], [1, 10, 1, 0.037296711977975315], [1, 10, 1, 0.061729083048426114], [1, 10, 1, 0.08364073635024263], [1, 10, 1, 0.09216999602957401], [1, 10, 1, 0.11052824151781825], [1, 10, 1, 0.10847854428531618], [1, 10, 1, 0.10247192602101635], [1, 10, 1, 0.11893503829634816], [1, 10, 1, 0.1300559267082629], [1, 10, 1, 0.11647261627067934], [1, 10, 1, 0.12628590519482097]]]