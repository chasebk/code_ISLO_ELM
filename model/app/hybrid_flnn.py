#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 13:57, 16/07/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from time import time
from permetrics.regression import Metrics
from numpy import reshape, add, matmul
from utils import math_util
from utils.io_util import save_to_csv_dict, save_to_csv, save_results_to_csv
from utils.visual_util import draw_predict
from utils.timeseries_util import *
from config import Config
from sklearn.metrics import mean_squared_error


class HybridFlnn:

    def __init__(self):
        self.solution = None
        self.list_loss = None
        self.filename = None

    def predict_using_model(self, data, model):
        hidd = getattr(math_util, self.activation)(add(matmul(data, model["w1"]), model["b1"]))
        y_pred = add(matmul(hidd, model["w2"]), model["b2"])
        return y_pred

    def predict_using_solution(self, data, solution):
        model = self.solution_to_network(solution)
        hidd = getattr(math_util, self.activation)(add(matmul(data, model["w1"]), model["b1"]))
        y_pred = add(matmul(hidd, model["w2"]), model["b2"])
        return y_pred

    def solution_to_network(self, solution):
        w1 = reshape(solution[:self.netsize["index_w1"]], (self.netsize["input"], self.netsize["hidden"]))
        b1 = reshape(solution[self.netsize["index_w1"]:self.netsize["index_b1"]], (-1, self.netsize["hidden"]))
        w2 = reshape(solution[self.netsize["index_b1"]: self.netsize["index_w2"]], (self.netsize["hidden"], self.netsize["output"]))
        b2 = reshape(solution[self.netsize["index_w2"]:], (-1, self.netsize["output"]))
        model = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
        self.weights = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
        return model

    def objective_function(self, solution):
        self.solution_to_network(solution)
        hidd = getattr(math_util, self.activation)(add(matmul(self.X_train, self.weights["w1"]), self.weights["b1"]))
        y_pred = add(matmul(hidd, self.weights["w2"]), self.weights["b2"])
        temp = mean_squared_error(self.y_train, y_pred)
        return temp

    def fit_model(self):
        pass

    # run a repeated experiment
    def experiment(self, optimizer, trials, datadict, series, epochs, activation, expand_func, verbose):
        time_prepare = time()
        self.optimizer = optimizer
        self.lag = datadict["lags"]
        self.test_size = int(datadict["test_percent"] * len(series.values))
        self.expand_func = expand_func
        self.verbose = verbose
        self.activation = activation
        self.netsize = {
            "input": self.lag * 5,
            "output": 1,
            "w": self.lag * 5,
            "b": 1,
            "index_w": self.lag * 5,
            "index_b": self.lag * 5 + 1,
        }
        self.problem_size = self.lag * 5 + 1
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
            train_trimmed = self.prepare_expansion(train_trimmed, self.expand_func)
            self.X_train, self.y_train = train_trimmed[:, 0:-1], train_trimmed[:, -1]
            self.fit_model()
            time_train = time() - time_train

            # forecast test dataset
            test_reshaped = self.prepare_expansion(test_scaled, self.expand_func)
            test_reshaped = test_reshaped[:, 0:-1]
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
            filename = f"{self.optimizer['name']}-{self.lag}-{self.expand_func}-{self.activation}-{self.filename}"

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



