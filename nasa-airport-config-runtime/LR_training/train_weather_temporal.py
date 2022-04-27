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

temporal_cross = 4


open_training_labels = pd.read_csv("../data/prescreened_train_labels.csv.bz2", parse_dates=["timestamp"], compression = "bz2")
airports = open_training_labels["airport"].unique().tolist()
for cur_cross in range(3,temporal_cross):
   airport_scores = []
   for air in airports:
      possible_labels = pd.read_csv(f"{air}_possibel_config")["0"].values.tolist()
      train = pd.read_csv("training_data.csv")
      train = train[train["airport"] == air]
      #feature_cols = ["temperature", "wind_speed", "wind_gust", "cloud_ceiling", "visibility", \
      #               "cloud", "lightning_prob", "precip","wind_direction_cos", "wind_direction_sin", "depart1", "deaprt2", "depart3", "depart4", \
      #                     "arrive1", "arrive2", "arrive3", "arrive4", "lookahead"]
      feature_cols = ["lookahead","wind_direction_cos","wind_direction_sin","depart3","arrive3","wind_speed","wind_gust","visibility"]
      for i in range(len(possible_labels)):
         feature_cols.append('cur_config_hot'+str(i))




      X = train.loc[:, feature_cols]
      y = train.actual_label
      temporal_split = X.shape[0]
      x_train = X.iloc[:int(temporal_split/temporal_cross*cur_cross)].copy()
      x_test = X.iloc[int(temporal_split/temporal_cross*cur_cross):int(temporal_split/temporal_cross*(cur_cross+1))].copy()
      y_train = y.iloc[:int(temporal_split/temporal_cross*cur_cross)].copy()
      y_test = y.iloc[int(temporal_split/temporal_cross*cur_cross):int(temporal_split/temporal_cross*(cur_cross+1))].copy()
      # Some weather data is missing :(
      imp = SimpleImputer(missing_values=np.nan, strategy='median')
      imp = imp.fit(x_train)
      x_train = imp.transform(x_train)
      imp = SimpleImputer(missing_values=np.nan, strategy='median')
      imp = imp.fit(x_test)
      x_test = imp.transform(x_test)



      print(f"Starting LR cross validation for {air}")
      #LRG = linear_model.LogisticRegression(penalty='l2',
      #   random_state = 0,solver = 'lbfgs', multi_class = 'multinomial', max_iter=100
      #).fit(x_train, y_train)

      #print(f"{air}Training error: "+str(LRG.score(X, y)))


      LRG = linear_model.LogisticRegression(penalty='l2',C=.33*cur_cross,solver = 'lbfgs', multi_class = 'multinomial', max_iter=5000000).fit(x_train, y_train)
      
      # LRG_boosted = GradientBoostingClassifier(n_estimators=70+cur_cross*10).fit(x_train, y_train)

      # a little bit of post processing
      predicted_probabilities = LRG.predict_proba(x_test).tolist()
      to_add = np.setdiff1d(np.array(range(len(possible_labels))),LRG.classes_)
      cur_classes = LRG.classes_
      print(cur_classes)
      for i in range(len(predicted_probabilities)):
         for j in to_add:
            predicted_probabilities[i].insert(j, 1e-5)
         x_p_other = predicted_probabilities[i][len(predicted_probabilities[i])-1]
         if x_p_other<.6:
            predicted_probabilities[i][len(predicted_probabilities[i])-1] = pow(x_p_other,.7)
      y_test_onehot = []
      for i in y_test:
         temp = [0]*(len(possible_labels))
         
         temp[int(i)] = 1
         y_test_onehot.append(temp)
      
      predicted_probabilities = normalize(np.array(predicted_probabilities), axis=1, norm="l1")

      score = log_loss(np.array(y_test_onehot).flatten(), predicted_probabilities.flatten())
      airport_scores.append(score)
      print(f"{air}scores for temporal{cur_cross}: "+ str(score))
      with open(f"{air}_trained_model_temporal{cur_cross}.pkl", "wb") as file:
         pickle.dump(LRG, file)
   print(f"mean score: {np.mean(np.array(airport_scores))}")

   #LRG = linear_model.LogisticRegression(penalty='none',
   #random_state = 0,solver = 'lbfgs', multi_class = 'multinomial', max_iter=1000000)
   #scores = cross_val_score(LRG, x_train, y_train, cv=2, scoring = make_scorer(log_loss, greater_is_better=True, needs_proba=True))
   #print(f"{air}scores for lbfgs no penalty: "+ str(scores))



