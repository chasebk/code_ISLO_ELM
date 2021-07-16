#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 16:46, 13/07/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from time import time
from permetrics.regression import Metrics
from numpy import reshape, add, matmul, dot
from numpy.linalg import pinv
from utils import math_util
from utils.io_util import save_to_csv_dict, save_to_csv, save_results_to_csv
from utils.visual_util import draw_predict
from utils.timeseries_util import *
from config import Config
from sklearn.metrics import mean_squared_error


class HybridElm:

    def __init__(self):
        self.solution = None
        self.list_loss = None
        self.filename = None

    def predict_using_model(self, data, model):
        hidden = getattr(math_util, model["activation"])(add(matmul(data, model["w1"]), model["b"]))
        y_pred = matmul(hidden, model["w2"])
        return y_pred.reshape(-1, 1)

    def predict_using_solution(self, data, solution):
        model = self.solution_to_network(solution)
        hidden = getattr(math_util, self.activation)(add(matmul(data, model["w1"]), model["b"]))
        y_pred = matmul(hidden, model["w2"])
        return y_pred.reshape(-1, 1)

    def solution_to_network(self, solution):
        w1 = reshape(solution[:self.netsize["index_w1"]], (self.netsize["input"], self.netsize["hidden"]))
        b = reshape(solution[self.netsize["index_w1"]:self.netsize["index_b"]], (-1, self.netsize["hidden"]))
        H = getattr(math_util, self.activation)(add(matmul(self.X_train, w1), b))
        w2 = dot(pinv(H), self.y_train)  # calculate weights between hidden and output layer
        model = {"w1": w1, "b": b, "w2": w2}
        self.weights = {"w1": w1, "b": b, "w2": w2}
        return model

    def objective_function(self, solution):
        self.solution_to_network(solution)
        hidden = getattr(math_util, self.activation)(add(matmul(self.X_train, self.weights["w1"]), self.weights["b"]))
        y_pred = matmul(hidden, self.weights["w2"])
        return mean_squared_error(self.y_train, y_pred)

    def fit_model(self):
        pass

    # run a repeated experiment
    def experiment(self, optimizer, trials, datadict, series, node, activation, verbose):
        time_prepare = time()
        self.optimizer = optimizer
        self.lag = datadict["lags"]
        self.test_size = int(datadict["test_percent"] * len(series.values))
        self.node = node
        self.verbose = verbose
        self.activation = activation
        self.netsize = {
            "input": self.lag,
            "hidden": self.node,
            "output": 1,
            "w1": self.lag * self.node,
            "b": self.node,
            "w2": self.node * 1,
            "index_w1": self.lag * self.node,
            "index_b": (self.lag + 1) * self.node,
        }
        self.problem_size = (self.lag + 1) * self.node
        self.lb = Config.LB * self.problem_size
        self.ub = Config.UB * self.problem_size
        # transform data to be stationary
        raw_values = series.values
        diff_values = difference(raw_values, 1)
        # transform data to be supervised learning
        supervised = timeseries_to_supervised(diff_values, self.lag)
        supervised_values = supervised.values[self.lag:, :]
        # split data into train and test-sets
        train, test = supervised_values[0:-self.test_size], supervised_values[-self.test_size:]
        # transform the scale of the data
        scaler, train_scaled, test_scaled = scale(train, test)

        time_prepare = time() - time_prepare
        # run experiment
        for trial in range(trials):
            time_train_test = time()

            # fit the model
            time_train = time()
            train_trimmed = train_scaled[2:, :]
            self.X_train, self.y_train = train_trimmed[:, 0:-1], train_trimmed[:, -1]
            self.fit_model()
            time_train = time() - time_train

            # forecast test dataset
            test_reshaped = test_scaled[:, 0:-1]
            output = self.predict_using_solution(test_reshaped, self.solution)
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
            test_true = array([raw_values[-self.test_size:]]).flatten()
            test_pred = array(test_pred).flatten()

            time_train_test = time() - time_train_test
            time_total = time_train_test + time_prepare

            ## Saving results
            # 1. Create path to save results
            path_general = f"{Config.DATA_RESULTS}/{datadict['dataname']}/{self.lag}-{datadict['test_percent']}-{trial}"
            filename = f"{self.optimizer['name']}-{node}-{self.activation}-{self.filename}"

            # 2. Saving performance of test set
            data = {"true": test_true, "predict": test_pred}
            save_to_csv_dict(data, f"predict-{filename}", f"{path_general}/{Config.FOL_RES_MODEL}")

            # 3. Save loss train to csv file
            data = [list(range(1, len(self.list_loss) + 1)), self.list_loss]
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


