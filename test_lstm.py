#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 09:59, 03/08/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from time import time
from pandas import read_csv
from permetrics.regression import Metrics
from keras.models import Sequential
from keras.layers import Dense, LSTM
from utils.io_util import save_to_csv_dict, save_to_csv, save_results_to_csv
from utils.visual_util import draw_predict
from utils.timeseries_util import *
from config import Config, Exp
from numpy import reshape

import os
import platform
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(2)  # matrix multiplication and reductions
tf.config.threading.set_inter_op_parallelism_threads(2)  # number of threads used by independent non-blocking operations

# if platform.system() == "Linux":  # Linux: "Linux", Mac: "Darwin", Windows: "Windows"
#     os.sched_setaffinity(0, {1})


def reshape_3d(data):
    return reshape(data, (data.shape[0], data.shape[1], 1))


def fit_model(train, batch_size, nb_epoch, neurons, verbose=2):
    #  The LSTM architecture
    X, y = train[:, 0:-1], train[:, -1]
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(None, 1), activation="relu"))
    model.add(Dense(units=1, activation="elu"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    loss = model.fit(reshape_3d(X), y, epochs=nb_epoch, batch_size=batch_size, verbose=verbose, shuffle=False)
    return model, loss


# run a repeated experiment
def experiment(datadict, series, epochs, neurons, verbose, path_general):
    time_total = time()
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

    # fit the model
    time_train = time()
    train_trimmed = train_scaled[2:, :]
    model, loss = fit_model(train_trimmed, batch_size, epochs, neurons, verbose)
    time_train = time() - time_train

    # forecast test dataset
    test_reshaped = test_scaled[:, 0:-1]
    output = model.predict(reshape_3d(test_reshaped), batch_size=batch_size)
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

    time_total = time() - time_total

    ## Saving results
    # 1. Create path to save results
    filename = f"LSTM-{neurons}-{epochs}-{batch_size}"

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
    return None


for dataname, datadict in Exp.LIST_DATASETS.items():
    # load dataset
    series = read_csv(f'{Config.DATA_APP}/{datadict["dataname"]}.csv', usecols=datadict["columns"])
    # experiment
    for trial in range(Exp.TRIAL):
        path_general = f"{Config.DATA_RESULTS}/{datadict['dataname']}/{datadict['lags']}-{datadict['test_percent']}-{trial}"
        experiment(datadict, series, Exp.EPOCH[0], Exp.NN_NET, Exp.VERBOSE, path_general)

