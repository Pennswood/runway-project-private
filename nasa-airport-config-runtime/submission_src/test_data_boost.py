import pandas as pd
from sklearn import datasets
from sklearn import linear_model
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

from sklearn.metrics import log_loss, make_scorer, accuracy_score
from sklearn.preprocessing import scale
from sklearn.impute import SimpleImputer, PolynominalFeatures
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from pathlib import Path
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

prediction_path = Path("/codeexecution/prediction.csv")

submission_format = pd.read_csv("/codeexecution/data/partial_submission_format.csv", parse_dates=["timestamp"])
test_data = pd.read_csv("./testing_data.csv")
airports = test_data["airport"].unique().tolist()
for air in airports:
    airport_test_data = test_data.copy()
    airport_test_data = airport_test_data[airport_test_data["airport"] == air]

    possible_labels = pd.read_csv(f"{air}_possibel_config")["0"].values.tolist()

    #feature_cols = ["temperature", "wind_speed", "wind_gust", "cloud_ceiling", "visibility", \
    #            "cloud", "lightning_prob", "precip","wind_direction_cos", "wind_direction_sin", "depart1", "deaprt2", "depart3", "depart4", \
    #                    "arrive1", "arrive2", "arrive3", "arrive4", "lookahead"]
    feature_cols = ["lookahead","wind_direction_cos","wind_direction_sin","depart3","arrive3","wind_speed","wind_gust","visibility"]

    X = airport_test_data.loc[:, feature_cols]
    for i in range(len(possible_labels)):
        feature_cols.append('cur_config_hot'+str(i))

    X = X.append(X.median(axis=0, skipna=False).fillna(0), ignore_index=True) # nothing is getting dropped!
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    X = imp.fit_transform(X)
    X = X[:-1] # remove the last row which is only used for making sure things don't get dropped


    # Polynomial kern
    polyF = PolynomialFeatures(degree=(2,2),include_bias=False,interaction_only=True)
    X = pd.DataFrame(polyF.fit_transform(X))
    
    # Put the linear/non-kern data back in! 
    X1 = airport_test_data.loc[:, feature_cols]
    X = pd.concat([X.reset_index(drop=True),X1.reset_index(drop=True)],axis=1)


    X = X.append(X.median(axis=0, skipna=False).fillna(0), ignore_index=True) # nothing is getting dropped!
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    X = imp.fit_transform(X)
    X = X[:-1] # remove the last row which is only used for making sure things don't get dropped

    
    model_file_stan = open(f"./{air}_standardize3_p.pkl", "rb")
    model_stan = pickle.load(model_file_stan)
    model_file_stan.close()
    X = pd.DataFrame(model_stan.transform(X),columns = X.columns)



    model_file = open(f"./{air}_trained_model_temporal3_p.pkl", "rb")
    model = pickle.load(model_file)
    model_file.close()

    predicted_probabilities = model.predict_proba(X).tolist()
    to_add = np.setdiff1d(np.array(range(len(possible_labels))),model.classes_)
    cur_classes = model.classes_
    for i in range(len(predicted_probabilities)):
        for j in to_add:
            predicted_probabilities[i].insert(j, 0.001)
        x_p_other = predicted_probabilities[i][len(predicted_probabilities[i])-1]
        if x_p_other<.6:
            predicted_probabilities[i][len(predicted_probabilities[i])-1] = pow(x_p_other,.65)


    predicted_probabilities = normalize(np.array(predicted_probabilities), axis=1, norm="l1")

    # Check this code!
    submission_format.loc[submission_format["airport"] == air, "active"] = predicted_probabilities.flatten()


submission_format.to_csv(prediction_path, date_format=DATETIME_FORMAT, index=False, float_format='%.15f')