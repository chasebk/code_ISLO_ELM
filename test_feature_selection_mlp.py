#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:36, 16/07/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import multiprocessing
from time import time
from pandas import read_csv
from permetrics.regression import Metrics
from keras.models import Sequential
from keras.layers import Dense
from utils.io_util import save_results_to_csv
from utils.timeseries_util import *
from config import Config, Exp
from sklearn.preprocessing import LabelEncoder
from model.app import mha


# fit an MLP network to training data
def fit_model(train, n_hidden, n_unit, batch_size, nb_epoch, activation, optimizer, verbose=2):
    X, y = train[:, 0:-1], train[:, -1]
    model = Sequential()
    model.add(Dense(n_unit, activation=activation, input_dim=X.shape[1]))  # input layers
    # Add each hidden layer.
    for i in range(0, n_hidden - 1):
        model.add(Dense(n_unit, activation=activation))
    model.add(Dense(1))  # Output layer.
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    loss = model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=verbose, shuffle=False)
    return model, loss.history["loss"]


def build_testcase(test_percentage, n_hidden, n_unit, batch_size, epoch, activation, optimizer):
    test_size = int(test_percentage * len(DATA_SERIES.values))

    # transform data to be stationary
    raw_values = DATA_SERIES.values
    diff_values = difference(raw_values, 1)
    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, LAGS)
    supervised_values = supervised.values[LAGS:, :]
    # split data into train and test-sets
    train, test = supervised_values[0:-test_size], supervised_values[-test_size:]
    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)

    # fit the model
    time_train = time()
    train_trimmed = train_scaled[2:, :]
    model, loss_train = fit_model(train_trimmed, n_hidden, n_unit, batch_size, epoch, activation, optimizer, Exp.VERBOSE)
    time_train = time() - time_train

    # forecast test dataset
    test_reshaped = test_scaled[:, 0:-1]
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

    ## Saving results
    # 1. Create path to save results
    filename = f"{test_percentage}-{n_hidden}-{n_unit}-{batch_size}-{epoch}-{activation}-{optimizer}"

    # 4. Calculate performance metrics and save it to csv file
    RM1 = Metrics(test_true, test_pred)
    list_paras = len(Config.METRICS_TEST_PHASE) * [{"decimal": 3}]
    mm1 = RM1.get_metrics_by_list_names(Config.METRICS_TEST_PHASE, list_paras)

    result_dict = {'filename': filename, 'time_train': time_train}
    for metric_name, value in mm1.items():
        result_dict[metric_name] = value
    save_results_to_csv(result_dict, f"{FILE_NAME}", f"{PATH_MODEL}")
    return result_dict


def objective_function(solution):
    # test_percentage = [0.1, 0.15, 0.2, 0.25, 0.3]
    # n_hidden = [1, 2, 3]
    # n_unit = list(range(5, 50))
    # batch_size = [32, 64, 128, 256, 512]
    # epoch = [750, 1000, 1250, 1500]
    # activation = ['elu', 'relu', 'sigmoid', 'tanh']
    # optimizer = ['adam', 'rmsprop', 'adadelta', "adagrad", "adamax", "nadam"]

    solution = solution.astype(int)
    test_percentage = 0.05 * (solution[0] + 1)      # lb = 1, ub = 5.99
    n_hidden = solution[1]                          # lb = 1, ub = 3.99
    n_unit = solution[2]                            # lb = 5, ub = 50.99
    batch_size = 2 ** solution[3]                   # lb = 6, ub = 9.99
    epoch = 5 * solution[4]                         # lb = 3, ub = 6.99
    act = solution[5]                               # lb = 0, ub = 3.99
    opt = solution[6]                               # lb = 0, ub = 5.99
    activation = ACT_ENCODER.inverse_transform([act])[0]
    optimizer = OPT_ENCODER.inverse_transform([opt])[0]
    result_dict = build_testcase(test_percentage, n_hidden, n_unit, batch_size, epoch, activation, optimizer)
    return result_dict["RMSE"]


OPT_ENCODER = LabelEncoder()
OPT_ENCODER.fit(['adadelta', 'adagrad', 'adam', 'adamax', 'nadam', 'rmsprop'])
ACT_ENCODER = LabelEncoder()
ACT_ENCODER.fit(['elu', 'relu', 'sigmoid', 'tanh'])
LB = [1, 1, 5, 6, 3, 1, 1]
UB = [5.99, 3.99, 50.99, 9.99, 6.99, 3.99, 5.99]
# LIST_MHAS = ["GA", "SADE", "SAP_DE", "SHADE", "L_SHADE", "WOA", "HI_WOA", "HGS", "COA", "LCBO", "CHIO", "OTWO", "OBL_HGSO", "SLO", "ISLO"]
LIST_MHAS = ["GA", "SAP_DE",]
MAX_GEN = 100
POP_SIZE = 50


def run_algorithm(name):
    global LAGS
    global DATA_SERIES
    global PATH_MODEL
    global FILE_NAME

    for dataname, datadict in Exp.LIST_DATASETS_FS.items():
        # load dataset
        DATA_SERIES = read_csv(f'{Config.DATA_APP}/{datadict["dataname"]}.csv', usecols=datadict["columns"])
        # experiment
        LAGS = datadict["lags"]
        PATH_MODEL = f"{Config.DATA_RESULTS}/{datadict['dataname']}/{datadict['lags']}"
        FILE_NAME = f"logging-{name}-{MAX_GEN}-{POP_SIZE}"

        md = getattr(mha, name)(objective_function, LB, UB, Exp.VERBOSE, MAX_GEN, POP_SIZE)
        solution, best_fit, list_loss = md.train()


if __name__ == '__main__':
    starttime = time()
    processes = []
    for algorithm in LIST_MHAS:
        p = multiprocessing.Process(target=run_algorithm, args=(algorithm,))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    print('That took: {} seconds'.format(time() - starttime))









