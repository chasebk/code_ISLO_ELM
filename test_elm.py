#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 16:12, 13/07/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from time import time
from pandas import read_csv
from permetrics.regression import Metrics
from numpy.random import uniform
from numpy.linalg import pinv
from numpy import add, matmul, dot
from sklearn.metrics import mean_squared_error
from utils.io_util import save_to_csv_dict, save_to_csv, save_results_to_csv
from utils.visual_util import draw_predict
from utils.timeseries_util import *
from config import Config, Exp
from utils import math_util


def fit_model(train, activation, neurons):
    """
        1. Random weights between input and hidden layer
        2. Calculate output of hidden layer
        3. Calculate weights between hidden and output layer based on matrix multiplication
    """
    X, y = train[:, 0:-1], train[:, -1:]
    input_size, output_size = X.shape[1], y.shape[1]
    w1 = uniform(size=[input_size, neurons])
    b = uniform(size=[1, neurons])
    H = getattr(math_util, activation)(add(matmul(X, w1), b))
    w2 = dot(pinv(H), y)
    model = {"w1": w1, "b": b, "w2": w2, "activation": activation}

    y_pred = matmul(H, model["w2"])
    loss_train = mean_squared_error(y, y_pred)
    return model, loss_train


def predict(model, data):
    hidden = getattr(math_util, model["activation"])(add(matmul(data, model["w1"]), model["b"]))
    y_pred = matmul(hidden, model["w2"])
    return y_pred


# run a repeated experiment
def experiment(trials, datadict, series, activation, neurons, verbose):
    time_prepare = time()
    lag = datadict["lags"]
    test_size = int(datadict["test_percent"] * len(series.values))

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
        model, loss_train = fit_model(train_trimmed, activation, neurons)
        time_train = time() - time_train

        # forecast test dataset
        test_reshaped = test_scaled[:, 0:-1]
        output = predict(model, test_reshaped)
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

        time_train_test = time() - time_train_test
        time_total = time_train_test + time_prepare

        ## Saving results
        # 1. Create path to save results
        path_general = f"{Config.DATA_RESULTS}/{datadict['dataname']}/{lag}-{datadict['test_percent']}-{trial}"
        filename = f"ELM-{neurons}-{activation}"

        # 2. Saving performance of test set
        data = {"true": test_true, "predict": test_pred}
        save_to_csv_dict(data, f"predict-{filename}", f"{path_general}/{Config.FOL_RES_MODEL}")

        # 3. Save loss train to csv file
        # data = [list(range(1, len(loss_train) + 1)), loss_train]
        # header = ["Epoch", "MSE"]
        # save_to_csv(data, header, f"loss-{filename}", f"{path_general}/{Config.FOL_RES_MODEL}")

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
    experiment(Exp.TRIAL, datadict, series, Exp.ACT, Exp.NN_NET, Exp.VERBOSE)

