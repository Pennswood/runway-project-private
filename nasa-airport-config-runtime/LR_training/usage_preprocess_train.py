import pandas as pd
from sklearn import datasets
from sklearn import linear_model
import datetime as dt
import numpy as np
import threading


def process_airports_departure_takeoff():
    opened = pd.read_csv("../data/prescreened_train_labels.csv.bz2", parse_dates=["timestamp"], compression = "bz2")
    airports = opened["airport"].unique().tolist()
    start_time = opened["timestamp"].min()
    end_time = opened["timestamp"].max()
    print("Processing departure and takeoff data.")
    running_threads = []
    for air in airports:
        newThread = threading.Thread(target=process_airport_departure_takeoff, args=(air,start_time,end_time,))
        newThread.start()
        running_threads.append(newThread)
    for r in running_threads:
        r.join()
    print("Finished processing departure and takeoff data.")


def process_airport_departure_takeoff(airport,start_time,end_time):
    timeBinsLanding = []
    timeBinsTakeOff = []
    curTime = (start_time)
    numBins = 15
    while curTime != (end_time):
        timeBinsLanding.append([0]*numBins)
        timeBinsTakeOff.append([0]*numBins)
        curTime += dt.timedelta(hours=.5)
    landing_file = f"../data/{airport}/{airport}_tfm_estimated_runway_arrival_time.csv.bz2"
    take_off_file = f"../data/{airport}/{airport}_etd.csv.bz2"
    chunksize=10**6
    landing_data = pd.read_csv(landing_file, parse_dates=["timestamp", "estimated_runway_arrival_time"], chunksize = chunksize)
    take_off_data = pd.read_csv(take_off_file, parse_dates=["timestamp", "estimated_runway_departure_time"], chunksize = chunksize)
    starting_i = 0
    for chunk in landing_data:
        for i in range(starting_i,len(timeBinsLanding)):
            curTime = start_time + dt.timedelta(hours=i*.5)
            getChunk = chunk.loc[chunk["timestamp"] <= curTime] # masking
            for j in range(numBins):

            
                getChunk1 = getChunk[(chunk["estimated_runway_arrival_time"] > curTime+dt.timedelta(hours=(j-2)*.5))]\
                                    [chunk["estimated_runway_arrival_time"]<=curTime+dt.timedelta(hours=(j-1)*.5)]
            
                timeBinsLanding[i][j] = (timeBinsLanding[i][j] + getChunk1["gufi"].unique().shape[0])

            if chunk["timestamp"].max() < curTime:
                starting_i = max(0,i - 1) 
                assert(getChunk.shape[0] == chunk.shape[0])
                break # chunk is behind, go to next chunk

        
        print(str(starting_i)+" out of "+str(len(timeBinsLanding)))
    pd.DataFrame({i: list(zip(*timeBinsLanding))[i] for i in range(len(list(zip(*timeBinsLanding))))}).to_csv(f"{airport}_landing_dist.csv")

    landing_data = pd.read_csv(landing_file, parse_dates=["timestamp", "estimated_runway_arrival_time"], chunksize = chunksize)
    take_off_data = pd.read_csv(take_off_file, parse_dates=["timestamp", "estimated_runway_departure_time"], chunksize = chunksize)
    starting_i = 0
    for chunk in take_off_data:
        for i in range(starting_i,len(timeBinsLanding)):
            curTime = start_time + dt.timedelta(hours=i*.5)
            getChunk = chunk.loc[chunk["timestamp"] <= curTime] # masking
            for j in range(numBins):

            
                getChunk1 = getChunk[(chunk["estimated_runway_departure_time"] > curTime+dt.timedelta(hours=(j-2)*.5))]\
                                    [chunk["estimated_runway_departure_time"]<=curTime+dt.timedelta(hours=(j-1)*.5)]
            
                timeBinsTakeOff[i][j] = (timeBinsTakeOff[i][j] + getChunk1["gufi"].unique().shape[0])

            if chunk["timestamp"].max() < curTime:
                starting_i = max(0,i - 1) 
                assert(getChunk.shape[0] == chunk.shape[0])
                break # chunk is behind, go to next chunk

        print(str(starting_i)+" out of "+str(len(timeBinsTakeOff)))
    pd.DataFrame({i: list(zip(*timeBinsTakeOff))[i] for i in range(len(list(zip(*timeBinsTakeOff))))}).to_csv(f"{airport}_take_off_dist.csv")

if __name__ == "__main__":
    process_airports_departure_takeoff()