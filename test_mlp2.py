#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 10:16, 06/07/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from pandas import DataFrame, Series, concat, read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from math import sqrt

from utils.visual_util import draw_predict
from utils.timeseries_util import *


# fit an MLP network to training data
def fit_model(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_dim=X.shape[1]))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=2, shuffle=False)
    return model


# run a repeated experiment
def experiment(repeats, series, epochs, lag, neurons, test_size, batch_size):
    # transform data to be stationary
    raw_values = series.values
    diff_values = difference(raw_values, 1)
    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, lag)
    # supervised = timeseries_to_supervised(raw_values, lag)
    supervised_values = supervised.values[lag:, :]
    # split data into train and test-sets
    train, test = supervised_values[0:-test_size], supervised_values[-test_size:]
    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)
    # run experiment
    error_scores = list()
    for r in range(repeats):
        # fit the model
        train_trimmed = train_scaled[2:, :]
        model = fit_model(train_trimmed, batch_size, epochs, neurons)
        # forecast test dataset
        test_reshaped = test_scaled[:, 0:-1]
        output = model.predict(test_reshaped, batch_size=batch_size)
        predictions = list()
        for i in range(len(output)):
            yhat = output[i, 0]
            X = test_scaled[i, 0:-1]
            # invert scaling
            yhat = invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
            # store forecast
            predictions.append(yhat)
        # report performance
        rmse = sqrt(mean_squared_error(raw_values[-test_size:], predictions))

        ## Drawing
        list_lines = [raw_values[-test_size:][:200], predictions[:200]]
        list_legends = ["Observed", "Predicted"]
        xy_labels = ["#Iteration", "CPU"]
        pathsave = "./data/app/results2"
        filename = f"mlp2-{dataname}-trial_{r}-{lag}-{neurons}-{test_size}-{batch_size}-{epochs}"
        exts = [".png", ".pdf"]
        draw_predict(list_lines, list_legends, xy_labels, "", filename, pathsave, exts, True)
        print('%d) Test RMSE: %.3f' % (r + 1, rmse))
        error_scores.append(rmse)
    return error_scores


dataname = "gg_cpu_5m"
# load dataset
series = read_csv(f'./data/app/clean/{dataname}.csv', usecols=[0])
dataset = series.values
# experiment
repeats = 3
results = DataFrame()
lags = [18, 36, 48]
neurons = 6
test_size = 1000
batch_size = 128
# vary training epochs
epoch = 1000
for lag in lags:
    results[str(lag)] = experiment(repeats, series, epoch, lag, neurons, test_size, batch_size)
# summarize results
print(results.describe())

