#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 09:37, 14/07/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from time import time
from pandas import read_csv
from permetrics.regression import Metrics
from keras.models import Sequential
from keras.layers import Dense

from utils.io_util import save_to_csv_dict, save_to_csv, save_results_to_csv
from utils.visual_util import draw_predict
from utils.timeseries_util import *
from config import Config, Exp
from utils import math_util
from numpy import concatenate


# fit an MLP network to training data
def fit_model(train, batch_size, nb_epoch, activation, verbose=2):
    X, y = train[:, 0:-1], train[:, -1]
    model = Sequential()
    model.add(Dense(1, activation=activation, input_dim=X.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    loss = model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=verbose, shuffle=False)
    return model, loss


def prepare_expansion(data, expansion_func):
    y_column = data[:, -1:]
    X_columns = data[:, 0:-1]
    X_expansion = getattr(math_util, f"expand_{expansion_func}")(X_columns)
    return concatenate((X_expansion, y_column), axis=1)


# run a repeated experiment
def experiment(trials, datadict, series, epochs, activation, expand_func, verbose):
    time_prepare = time()
    lag = datadict["lags"]
    test_size = int(datadict["test_percent"] * len(series.values))
    batch_size = datadict["batch_size"]

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

    time_prepare = time() - time_prepare
    # run experiment
    for trial in range(trials):
        time_train_test = time()

        # fit the model
        time_train = time()
        train_trimmed = train_scaled[2:, :]
        train_trimmed = prepare_expansion(train_trimmed, expand_func)
        model, loss = fit_model(train_trimmed, batch_size, epochs, activation, verbose)
        time_train = time() - time_train

        # forecast test dataset
        test_reshaped = prepare_expansion(test_scaled, expand_func)
        test_reshaped = test_reshaped[:, 0:-1]
        output = model.predict(test_reshaped, batch_size=batch_size)
        test_pred = list()
        for i in range(len(output)):
            yhat = output[i, 0]
            X = test_scaled[i, 0:-1]
            # invert scaling
            yhat = invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
            # store forecast
            test_pred.append(yhat)
        test_true = array([raw_values[-test_size:]]).flatten()
        test_pred = array(test_pred).flatten()
        loss_train = loss.history["loss"]

        time_train_test = time() - time_train_test
        time_total = time_train_test + time_prepare

        ## Saving results
        # 1. Create path to save results
        path_general = f"{Config.DATA_RESULTS}/{datadict['dataname']}/{lag}-{datadict['test_percent']}-{trial}"
        filename = f"FLNN-{lag}-{expand_func}-{epochs}-{batch_size}-{activation}"

        # 2. Saving performance of test set
        data = {"true": test_true, "predict": test_pred}
        save_to_csv_dict(data, f"predict-{filename}", f"{path_general}/{Config.FOL_RES_MODEL}")

        # 3. Save loss train to csv file
        data = [list(range(1, len(loss_train) + 1)), loss_train]
        header = ["Epoch", "MSE"]
        save_to_csv(data, header, f"loss-{filename}", f"{path_general}/{Config.FOL_RES_MODEL}")

        # 4. Calculate performance metrics and save it to csv file
        RM1 = Metrics(test_true, test_pred)
        list_paras = len(Config.METRICS_TEST_PHASE) * [{"decimal": 3}]
        mm1 = RM1.get_metrics_by_list_names(Config.METRICS_TEST_PHASE, list_paras)

        item = {'filename': filename, 'time_train': time_train, 'time_total': time_total}
        for metric_name, value in mm1.items():
            item[metric_name] = value
        save_results_to_csv(item, f"metrics-{filename}", f"{path_general}/{Config.FOL_RES_MODEL}")

        # 5. Saving performance figure
        list_lines = [test_true[200:400], test_pred[200:400]]
        list_legends = ["Observed", "Predicted"]
        xy_labels = ["#Iteration", datadict["datatype"]]
        exts = [".png", ".pdf"]
        draw_predict(list_lines, list_legends, xy_labels, "", filename, f"{path_general}/{Config.FOL_RES_VISUAL}", exts, verbose)


for dataname, datadict in Exp.LIST_DATASETS.items():
    # load dataset
    series = read_csv(f'{Config.DATA_APP}/{datadict["dataname"]}.csv', usecols=datadict["columns"])
    # experiment
    results = DataFrame()
    for expand in Exp.EXPANDS:
        for act in Exp.ACTIVATIONS:
            experiment(Exp.TRIAL, datadict, series, Exp.EPOCH[0], act, expand, Exp.VERBOSE)

