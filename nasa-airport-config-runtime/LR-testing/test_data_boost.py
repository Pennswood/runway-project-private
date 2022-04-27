import pandas as pd
from sklearn import datasets
from sklearn import linear_model
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

from sklearn.metrics import log_loss, make_scorer, accuracy_score
from sklearn.preprocessing import scale
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from pathlib import Path
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

prediction_path = Path("./prediction.csv")

submission_format = pd.read_csv("../data/partial_submission_format.csv", parse_dates=["timestamp"])
test_data = pd.read_csv("./testing_data.csv")
airports = test_data["airport"].unique().tolist()
for air in airports:
    airport_test_data = test_data.copy()
    airport_test_data = airport_test_data[airport_test_data["airport"] == air]

    possible_labels = pd.read_csv(f"{air}_possibel_config")["0"].values.tolist()

    feature_cols = ["temperature", "wind_speed", "wind_gust", "cloud_ceiling", "visibility", \
                "cloud", "lightning_prob", "precip","wind_direction_cos", "wind_direction_sin", "depart1", "deaprt2", "depart3", "depart4", \
                        "arrive1", "arrive2", "arrive3", "arrive4", "lookahead"]
    for i in range(len(possible_labels)):
        feature_cols.append('cur_config_hot'+str(i))

    X = airport_test_data.loc[:, feature_cols]

    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-10)
    imp = imp.fit(X)
    X = imp.transform(X)

    model_file = open(f"{air}_trained_model_LR.pkl", "rb")
    model = pickle.load(model_file)
    model_file.close()

    predicted_probabilities = model.predict_proba(X).tolist()
    to_add = np.setdiff1d(np.array(range(len(possible_labels))),model.classes_)
    cur_classes = model.classes_
    for i in range(len(predicted_probabilities)):
        for j in to_add:
            predicted_probabilities[i].insert(j, 1e-6)


    predicted_probabilities = normalize(np.array(predicted_probabilities), axis=1, norm="l1")

    # Check this code!
    submission_format.loc[submission_format["airport"] == air, "active"] = predicted_probabilities.flatten()


submission_format.to_csv(prediction_path, date_format=DATETIME_FORMAT, index=False, float_format='%.15f')