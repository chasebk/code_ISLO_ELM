#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 14:47, 16/07/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import multiprocessing
from time import time
from pandas import read_csv
from sklearn.model_selection import ParameterGrid
from config import Config, Exp
from model.app import mha_flnn


def setting_and_running(optimizer):
    print(f"Start running: {optimizer['name']}")
    for dataname, datadict in Exp.LIST_DATASETS.items():
        # load dataset
        series = read_csv(f'{Config.DATA_APP}/{datadict["dataname"]}.csv', usecols=datadict["columns"])
        # experiment
        parameters_grid = list(ParameterGrid(optimizer["param_grid"]))
        for mha_paras in parameters_grid:
            hybridmodel = getattr(mha_flnn, optimizer["class"])(mha_paras)
            for activation in Exp.FLNN_ACTIVATIONS:
                for expand in Exp.EXPANDS:
                    hybridmodel.experiment(optimizer, Exp.TRIAL, datadict, series, Exp.EPOCH[0], activation, expand, Exp.VERBOSE)


if __name__ == '__main__':
    starttime = time()
    processes = []
    for optimizer in Exp.FLNN_OPTIMIZERS:
        p = multiprocessing.Process(target=setting_and_running, args=(optimizer,))
        processes.append(p)
        p.start()
    for process in processes:
        process.join()
    print('That took: {} seconds'.format(time() - starttime))
