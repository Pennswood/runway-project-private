# runway-project-private
(Keep private, do not share with others until after April 25)

To get data, go to https://www.drivendata.org/competitions/89/competition-nasa-airport-configuration/data/, download, and unzip the data. Also download the training labels and place it into the data folder.

To build and test the model, run train_weather_temporal.py for training models that can be tested to give a score, or run train_weather_temporal_p.py to create teh final models that were used in the competition. These files were not included due to size limits.

Make sure to move the pkl files into LR_testing and submission_src for building.

A summary with block diagrams, methodology, and "little tweaks", and hyperparameter tuning is captured in the PowerPoint (https://github.com/Pennswood/runway-project-private/blob/main/weather%20for%20predicting%20airport%20configuration.pptx). 
