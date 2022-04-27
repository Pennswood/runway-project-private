import pandas as pd
from sklearn import datasets
from sklearn import linear_model
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

from sklearn.metrics import log_loss, make_scorer, accuracy_score
from sklearn.preprocessing import scale
from sklearn.impute import SimpleImputer
import pickle


airport_scores = []
open_training_labels = pd.read_csv("../data/open_train_labels.csv.bz2", parse_dates=["timestamp"], compression = "bz2")
airports = open_training_labels["airport"].unique().tolist()
for air in airports:
   possible_labels = pd.read_csv(f"{air}_possibel_config")["0"].values.tolist()
   train = pd.read_csv("training_data_kern.csv")
   train = train[train["airport"] == air]
   feature_cols = ["temperature", "wind_speed", "wind_gust", "cloud_ceiling", "visibility", \
                  "cloud", "lightning_prob", "precip","wind_direction_cos", "wind_direction_sin","gust_dir_cos", "gust_dir_sin", "speed_dir_cos","speed_dir_sin",  "depart1", "deaprt2", "depart3", "depart4", \
                        "arrive1", "arrive2", "arrive3", "arrive4", "lookahead"]
   for i in range(len(possible_labels)):
      feature_cols.append('cur_config_hot'+str(i))

   kerned_features = []
   for i in feature_cols:
      if str(i) != "airport" and str(i) != "actual_label":
         kerned_features.append(str(i)+"_lookahead_kern")
   
   feature_cols = feature_cols + kerned_features



   X = train.loc[:, feature_cols]
   y = train.actual_label
   x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
   # Some weather data is missing :(
   imp = SimpleImputer(missing_values=np.nan, strategy='median')
   imp = imp.fit(x_train)
   x_train = imp.transform(x_train)
   x_test = imp.transform(x_test)



   print(f"Starting LR cross validation for {air}")
   #LRG = linear_model.LogisticRegression(penalty='l2',
   #   random_state = 0,solver = 'lbfgs', multi_class = 'multinomial', max_iter=100
   #).fit(x_train, y_train)

   #print(f"{air}Training error: "+str(LRG.score(X, y)))


   LRG = linear_model.LogisticRegression(penalty='l2', C = 2, solver = 'lbfgs', multi_class = 'multinomial', max_iter=1000000
   ).fit(x_train, y_train)

   # a little bit of post processing
   predicted_probabilities = LRG.predict_proba(x_test).tolist()
   to_add = np.setdiff1d(np.array(range(len(possible_labels))),LRG.classes_)
   cur_classes = LRG.classes_
   for i in range(len(predicted_probabilities)):
      for j in to_add:
         predicted_probabilities[i].insert(j, 1e-8)
   y_test_onehot = []
   for i in y_test:
      temp = [0]*(len(possible_labels))
      
      temp[int(i)] = 1
      y_test_onehot.append(temp)
         
   predicted_probabilities = normalize(np.array(predicted_probabilities), axis=1, norm="l1")

   score = log_loss(np.array(y_test_onehot).flatten(), predicted_probabilities.flatten())
   airport_scores.append(score)
   print(f"{air}scores for lbfgs l2: "+ str(score))
   with open(f"{air}_trained_model_kerned.pkl", "wb") as file:
      pickle.dump(LRG, file)
   

   #LRG = linear_model.LogisticRegression(penalty='none',
   #random_state = 0,solver = 'lbfgs', multi_class = 'multinomial', max_iter=1000000)
   #scores = cross_val_score(LRG, x_train, y_train, cv=2, scoring = make_scorer(log_loss, greater_is_better=True, needs_proba=True))
   #print(f"{air}scores for lbfgs no penalty: "+ str(scores))

print(f"mean score: {np.mean(np.array(airport_scores))}")

