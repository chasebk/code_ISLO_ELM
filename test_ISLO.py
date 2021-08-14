#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 16:11, 14/08/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%


import multiprocessing
from pathlib import Path
from config import Config
from model import benchmark
from pandas import DataFrame
from time import time
from utils.IOUtil import save_results_to_csv

TRIALS = 20
PROBLEM_SIZE = 30
LB = [-100] * PROBLEM_SIZE
UB = [100] * PROBLEM_SIZE
VERBOSE = False
EPOCH = 1000
POP_SIZE = 50
LIST_FUNCTIONS = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20"]
LIST_MHAS = ["ImprovedSLO2"]


def run_algorithm(name):
    path_error = f"{Config.BENCHMARK_ERROR}/{name}/"
    Path(path_error).mkdir(parents=True, exist_ok=True)

    ## Run model
    for id_paras, func_name in enumerate(LIST_FUNCTIONS):
        error_full = {}
        error_columns = []
        for id_trial in range(TRIALS):
            time_start = time()
            md = getattr(benchmark, name)(getattr(benchmark, func_name), LB, UB, VERBOSE, EPOCH, POP_SIZE)
            _, best_fit, list_loss = md.train()
            temp = f"trial_{str(id_trial)}"
            error_full[temp] = list_loss
            error_columns.append(temp)
            time_end = time() - time_start
            item = {'function': func_name, 'time': time_end, 'trial': id_trial, 'fit': best_fit}
            save_results_to_csv(item, f"{PROBLEM_SIZE}D_{name}_best_fit", Config.BENCHMARK_BEST_FIT)

        df = DataFrame(error_full, columns=error_columns)
        df.to_csv(f"{path_error}/{PROBLEM_SIZE}D_{name}_{func_name}_error.csv", header=True, index=False)


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

