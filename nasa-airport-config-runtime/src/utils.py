from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


hyperParams = {('katl', 30): [0.25, 9], ('katl', 60): [0.25, 6], ('katl', 90): [0.25, 3], ('katl', 120): [0.25, 3], ('katl', 150): [0.25, 3], ('katl', 180): [0.25, 3], ('katl', 210): [0.25, 3], ('katl', 240): [0.25, 3], ('katl', 270): [1, 3], ('katl', 300): [1, 3], ('katl', 330): [1, 3], ('katl', 360): [1, 3], ('kclt', 30): [0.25, 9], ('kclt', 60): [0.25, 6], ('kclt', 90): [0.25, 3], ('kclt', 120): [0.25, 3], ('kclt', 150): [0.25, 3], ('kclt', 180): [0.25, 3], ('kclt', 210): [1, 3], ('kclt', 240): [1, 3], ('kclt', 270): [1.5, 3], ('kclt', 300): [1.5, 3], ('kclt', 330): [1.5, 3], ('kclt', 360): [1.5, 3], ('kden', 30): [0.25, 9], ('kden', 60): [1, 3], ('kden', 90): [1, 3], ('kden', 120): [1.5, 3], ('kden', 150): [1.5, 3], ('kden', 180): [1.5, 3], ('kden', 210): [1.5, 3], ('kden', 240): [1.5, 3], ('kden', 270): [1.5, 3], ('kden', 300): [1.5, 3], ('kden', 330): [1.5, 3], ('kden', 360): [1.5, 3], ('kdfw', 30): [0.25, 9], ('kdfw', 60): [0.25, 6], ('kdfw', 90): [0.25, 3], ('kdfw', 120): [0.25, 3], ('kdfw', 150): [0.25, 3], ('kdfw', 180): [1, 3], ('kdfw', 210): [1, 3], ('kdfw', 240): [1, 3], ('kdfw', 270): [1.5, 3], ('kdfw', 300): [1.5, 3], ('kdfw', 330): [1.5, 3], ('kdfw', 360): [1.5, 3], ('kjfk', 30): [0.25, 9], ('kjfk', 60): [0.25, 9], ('kjfk', 90): [0.25, 6], ('kjfk', 120): [0.25, 3], ('kjfk', 150): [0.25, 3], ('kjfk', 180): [0.25, 3], ('kjfk', 210): [0.25, 3], ('kjfk', 240): [0.25, 3], ('kjfk', 270): [0.25, 3], ('kjfk', 300): [1, 3], ('kjfk', 330): [1, 3], ('kjfk', 360): [1, 3], ('kmem', 30): [0.25, 9], ('kmem', 60): [0.25, 3], ('kmem', 90): [0.25, 3], ('kmem', 120): [1.5, 3], ('kmem', 150): [1.5, 3], ('kmem', 180): [1.5, 3], ('kmem', 210): [1.5, 3], ('kmem', 240): [1.5, 3], ('kmem', 270): [1.5, 3], ('kmem', 300): [1.5, 3], ('kmem', 330): [1.5, 3], ('kmem', 360): [1.5, 3], ('kmia', 30): [0.25, 9], ('kmia', 60): [0.25, 6], ('kmia', 90): [0.25, 6], ('kmia', 120): [0.25, 3], ('kmia', 150): [0.25, 3], ('kmia', 180): [1, 3], ('kmia', 210): [1, 3], ('kmia', 240): [1, 3], ('kmia', 270): [1, 3], ('kmia', 300): [1.5, 3], ('kmia', 330): [1.5, 3], ('kmia', 360): [1.5, 3], ('kord', 30): [0.25, 6], ('kord', 60): [1, 3], ('kord', 90): [1, 3], ('kord', 120): [1.5, 3], ('kord', 150): [1.5, 3], ('kord', 180): [1.5, 3], ('kord', 210): [1.5, 3], ('kord', 240): [1.5, 3], ('kord', 270): [1.5, 3], ('kord', 300): [1.5, 3], ('kord', 330): [1.5, 3], ('kord', 360): [1.5, 3], ('kphx', 30): [0.25, 9], ('kphx', 60): [0.25, 6], ('kphx', 90): [0.25, 3], ('kphx', 120): [0.25, 3], ('kphx', 150): [0.25, 3], ('kphx', 180): [0.25, 3], ('kphx', 210): [0.25, 3], ('kphx', 240): [1, 3], ('kphx', 270): [1, 3], ('kphx', 300): [1, 3], ('kphx', 330): [1.5, 3], ('kphx', 360): [1.5, 3], ('ksea', 30): [0.25, 9], ('ksea', 60): [0.25, 6], ('ksea', 90): [0.25, 6], ('ksea', 120): [0.25, 3], ('ksea', 150): [0.25, 3], ('ksea', 180): [0.25, 3], ('ksea', 210): [0.25, 3], ('ksea', 240): [0.25, 3], ('ksea', 270): [0.25, 3], ('ksea', 300): [0.25, 3], ('ksea', 330): [0.25, 3], ('ksea', 360): [0.25, 3]}

#"ad hoc"
airportChangeDist = [305.75434295696164, 324.1129672893487, 360.0033157049654, 419.7161412026453, 480.96472685820254, 543.6522834849023, 573.38931438297, 573.8467501566496, 543.246278514454, 511.84695761147515, 484.29743094717946, 474.0203221053701, 466.98484309641145, 458.586462090806, 433.14286440075193, 401.0643872038477, 365.2567359311758, 347.96592660332936, 358.39082795644293, 415.9948517333062, 503.7832139409643, 599.8954935731341, 661.4446307304103, 680.2327897170146, 646.1455232095378, 598.1769208623347, 530.7609612355842, 470.9293637487511, 406.35196192992265, 355.46693424105405, 317.3607512404952, 294.1946518992701, 276.3987874477129, 273.19719216244135, 270.4532676251926, 277.47304780775283, 272.5073921658283, 282.2760419312774, 294.2573625294247, 315.8128588121729, 312.322050331081, 297.8461468441464, 268.61172818337315, 249.3823186040407, 232.57994605116087, 222.22219483500987, 212.8579791121568, 208.60830997481963]


def read_airport_configs(airport_directory: Path) -> Tuple[str, pd.DataFrame]:
    """Reads the airport configuration features for a given airport data directory."""
    airport_code = airport_directory.name
    #TODO CHANGE HERE
    filename = f"{airport_code}_airport_config.csv.bz2"
    filepath = airport_directory / filename
    airport_config_df = pd.read_csv(filepath, parse_dates=["timestamp"])
    return airport_code, airport_config_df


def make_prediction(
    airport_config_df_map: Dict[str, pd.DataFrame],
    pred_frame: pd.DataFrame,
    hedge: float = 1,
    weight: float = 8,
    discount_factor: float = 0.89,
    temporal_bias = 0
) -> pd.Series:
    # start with a uniform distribution
    uniform = make_uniform(pred_frame) * hedge
    predictive_distribution = pd.DataFrame({"uniform": uniform})

    # select the data we're allowed to use
    first = pred_frame.iloc[0]
    airport_code, timestamp, lookahead, _, _ = first
    airport_config_df = airport_config_df_map[airport_code]
    # if there is no data, return the uniform probability
    if len(airport_config_df) == 0:
        return uniform / uniform.sum()
    current, subset = censor_data(airport_config_df, timestamp)
    if len(subset) == 0:
        return uniform / uniform.sum()

    # make the distribution of past configurations
    config_dist = make_config_dist(airport_code, subset, normalize=True)
    predictive_distribution["config_dist"] = config_dist.reindex(
        predictive_distribution.index
    ).fillna(0)
    other = config_dist.sum() - predictive_distribution.config_dist.sum()
    predictive_distribution.loc[f"{airport_code}:other", "config_dist"] += other

    # put some extra weight on the current configuration (or `other`)
    current_key = f"{airport_code}:{current}"
    if current_key not in pred_frame.config.values:
        current_key = f"{airport_code}:other"
    # discount = pow(discount_factor, lookahead/30)

    # Too adhoc.
    # discount = (1-probChange(timestamp, lookahead, temporal_bias))
    
    discount = 1
    predictive_distribution["current"] = 0  # initalize a column of zeros
    predictive_distribution.loc[current_key, "current"] = weight * discount

    # combine the components and normalize the result
    mixture = predictive_distribution.sum(axis=1)

    predictive_distribution["mixture"] = mixture / mixture.sum()

    return predictive_distribution.mixture

def probChange(timestamp, lookahead, temporal_bias):
    global airportChangeDist
    time = str(timestamp).split()[1].split(":")
    time = (int(time[0])*60+int(time[1]))/30
    lookahead_time = time + lookahead/30
    if lookahead_time < 24*60/30:
        lookahead_time = lookahead_time % (int(24*60/30))
    output = 0
    biasedChangeDist = [a - temporal_bias for a in airportChangeDist]
    for a in biasedChangeDist:
        assert(a > 0)
    for i in range(int(24*60/30)):
        if int(time) >=int(lookahead_time) and int(time) < i and int(lookahead_time) > i:
            output += biasedChangeDist[i]
        if int(time) < int(lookahead_time) and int(time) < i or int(lookahead_time) < i: # loopback to the next day
            output += biasedChangeDist[i]
        
    output = output/sum(biasedChangeDist)
    #output /=365
    return output
    

def make_all_predictions(
    airport_config_df_map: Dict[str, pd.DataFrame], predictions: pd.DataFrame
):
    global hyperParams
    """Predicts airport configuration for all of the prediction frames in a table."""
    all_preds = []
    grouped = predictions.groupby(["airport", "timestamp", "lookahead"], sort=False)
    for key, pred_frame in tqdm(grouped):
        airport, timestamp, lookahead = key
        airport, timestamp, lookahead = key
        hyp = hyperParams[(airport,lookahead)]
        pred_dist = make_prediction(
            airport_config_df_map, pred_frame, hedge = hyp[0], weight = hyp[1])
        assert np.array_equal(pred_dist.index.values, pred_frame["config"].values)
        all_preds.append(pred_dist.values)

    predictions["active"] = np.concatenate(all_preds)

def make_all_predictions_test(
    airport_config_df_map: Dict[str, pd.DataFrame], predictions: pd.DataFrame, hedge: float, weight: float
):
    """Predicts airport configuration for all of the prediction frames in a table."""
    all_preds = []
    grouped = predictions.groupby(["airport", "timestamp", "lookahead"], sort=False)
    for key, pred_frame in tqdm(grouped):
        airport, timestamp, lookahead = key
        pred_dist = make_prediction(
            airport_config_df_map, pred_frame, hedge = hedge, weight = weight)
        assert np.array_equal(pred_dist.index.values, pred_frame["config"].values)
        all_preds.append(pred_dist.values)

    predictions["active"] = np.concatenate(all_preds)




def make_uniform(pred_frame: pd.DataFrame) -> pd.Series:
    indices = pred_frame["config"].values
    uniform = pd.Series(1, index=indices)
    uniform /= uniform.sum()
    return uniform


def make_config_dist(
    airport_code: str, airport_config_df: pd.DataFrame, normalize: bool = False
) -> pd.Series:
    config_timecourse = (
        airport_config_df.set_index("timestamp")
        .airport_config.resample("15min")
        .ffill()
        .dropna()
    )
    config_dist = config_timecourse.value_counts()
    if normalize:
        config_dist /= config_dist.sum()

    # prepend the airport code to the configuration strings
    prefix = pd.Series(f"{airport_code}:", index=config_dist.index)
    config_dist.index = prefix.str.cat(config_dist.index)
    return config_dist

def censor_data(
    airport_config_df: pd.DataFrame, timestamp: pd.Timestamp
) -> Tuple[str, pd.DataFrame]:
    mask = airport_config_df["timestamp"] <= timestamp
    subset = airport_config_df[mask]
    if subset.shape[0] > 0:
        current = subset.iloc[-1].airport_config
    else:
        current = airport_config_df.iloc[-1].airport_config
    return current, subset