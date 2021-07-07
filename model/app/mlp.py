#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 13:43, 06/07/2021                                                               %
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

class MLP:

    # fit an MLP network to training data
    def fit_model(self, train, batch_size, nb_epoch, neurons, activation, optimizer, verbose):
        X, y = train[:, 0:-1], train[:, -1]
        model = Sequential()
        model.add(Dense(neurons, activation=activation, input_dim=X.shape[1]))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=verbose, shuffle=False)
        self.model = model 
        return model

    # run a repeated experiment
    def experiment(self, trials, series, epochs, lag, neurons, test_size, batch_size, optimizer, verbose):
        # transform data to be stationary
        raw_values = series.values
        diff_values = difference(raw_values, 1)
        # transform data to be supervised learning
        supervised = timeseries_to_supervised(diff_values, lag)
        supervised_values = supervised.values[lag:, :]
        # split data into train and test-sets
        train, test = supervised_values[0:-test_size], supervised_values[-test_size:]
        # transform the scale of the data
        scaler, train_scaled, test_scaled = scale(train, test)
        # run experiment
        error_scores = list()
        for r in range(trials):
            # fit the model
            train_trimmed = train_scaled[2:, :]
            model = self.fit_model(train_trimmed, batch_size, epochs, neurons, optimizer, verbose)
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
            list_lines = [raw_values[-test_size:][:100], predictions[:100]]
            list_legends = ["Observed", "Predicted"]
            xy_labels = ["#Iteration", "CPU"]
            pathsave = "./data/app/results"
            filename = f"mlp2-{dataname}-{lag}-{neurons}-{test_size}-{batch_size}-{epochs}"
            exts = [".png", ".pdf"]
            draw_predict(list_lines, list_legends, xy_labels, "", filename, pathsave, exts, True)
            print('%d) Test RMSE: %.3f' % (r + 1, rmse))
            error_scores.append(rmse)
        return error_scores

