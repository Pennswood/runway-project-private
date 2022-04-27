import pandas as pd
from sklearn import datasets
from sklearn import linear_model
import datetime as dt
import numpy as np
import math

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


training_data = pd.read_csv("./training_data.csv")
training_data["gust_dir_cos"] = training_data["wind_gust"]*training_data["wind_direction_cos"]
training_data["gust_dir_sin"] = training_data["wind_gust"]*training_data["wind_direction_sin"]
training_data["speed_dir_cos"] = training_data["wind_speed"]*training_data["wind_direction_cos"]
training_data["speed_dir_sin"] = training_data["wind_speed"]*training_data["wind_direction_sin"]
for i in training_data:
    if str(i) != "airport" and str(i) != "actual_label":
        training_data[str(i)+"_lookahead_kern"] = training_data[str(i)]*training_data["lookahead"]/30



training_data.to_csv('training_data_kern.csv')