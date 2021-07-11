#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 22:11, 31/05/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from os.path import abspath, dirname

basedir = abspath(dirname(__file__))


class Config:
    DATA_DIRECTORY = f'{basedir}/data'

    BENCHMARK_RESULTS = f'{DATA_DIRECTORY}/benchmark'
    BENCHMARK_ERROR = f'{BENCHMARK_RESULTS}/error'
    BENCHMARK_BEST_FIT = f'{BENCHMARK_RESULTS}/best_fit'
    BM_STT_MIN = f"{BENCHMARK_RESULTS}/min.csv"
    BM_STT_MAX = f"{BENCHMARK_RESULTS}/max.csv"
    BM_STT_MEAN = f"{BENCHMARK_RESULTS}/mean.csv"
    BM_STT_STD = f"{BENCHMARK_RESULTS}/std.csv"
    BM_STT_CV = f"{BENCHMARK_RESULTS}/cv.csv"
    BM_STATISTICS = f"{BENCHMARK_RESULTS}/statistics.csv"
    BM_STT_COLS = ["function", "mha", "time-min", "time-max", "time-cv", "time-mean", "time-std",
                   "fit-min", "fit-max", "fit-cv", "fit-mean", "fit-std"]


    DATA_APP = f'{DATA_DIRECTORY}/app/clean'
    DATA_RESULTS = f'{DATA_DIRECTORY}/app/results1'
    FOL_RES_VISUAL = "visualize"
    FOL_RES_MODEL = "model"

    METRICS_TEST_PHASE = ["MAE", "RMSE", "R", "R2s", "MAPE", "NSE", "KGE", "PCD", "KLD", "VAF", "A10", "A20"]
    LB = [-3]
    UB = [3]


class Exp:
    NN_NET = 20     # The number hidden neuron of the network for traditional MLP
    NN_HYBRID = 3   # For hybrid models
    ACT = "relu"    # Activation function for hybrid models

    VERBOSE = 0
    TRIAL = 10

    EPOCH = [1000]
    POP_SIZE = [50]

    ## Evolutionary-based group
    ga_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "pc": [0.85],  # crossover probability
        "pm": [0.05]  # mutation probability
    }

    de_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "wf": [0.85],  # weighting factor
        "cr": [0.8],  # crossover rate
    }
    jade_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "miu_f": [0.5],  # weighting factor
        "miu_cr": [0.5],  # crossover rate
        "pp": [0.1],
        "cc": [0.1]
    }

    ## Swarm-based group
    pso_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "c1": [1.2],
        "c2": [1.2],
        "w_min": [0.4],
        "w_max": [0.9],
    }
    clpso_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
        "c_local": [1.2],
        "w_min": [0.4],
        "w_max": [0.9],
    }

    slo_paras = islo_paras = {
        "epoch": EPOCH, "pop_size": POP_SIZE,
    }


    LIST_DATASETS = {
        "cpu": {
            "datatype": "CPU",
            "dataname": "gg_cpu_5m",
            "columns": [0],
            "lags": 36,
            "test_percent": 0.2,
            "batch_size": 128,
        },
        "ram": {
            "datatype": "RAM",
            "dataname": "gg_ram_5m",
            "columns": [1],
            "lags": 36,
            "test_percent": 0.2,
            "batch_size": 128,
        },
        "it_eu": {
            "datatype": "Internet Traffic EU (in Megabyte)",
            "dataname": "it_eu_5m",
            "columns": [0],
            "lags": 41,
            "test_percent": 0.2,
            "batch_size": 256,
        },
        "it_uk": {
            "datatype": "Internet Traffic UK (in Byte)",
            "dataname": "it_uk_5m",
            "columns": [0],
            "lags": 43,
            "test_percent": 0.2,
            "batch_size": 256,
        }
    }

    OPTIMIZERS = [
        {"name": "GA-MLP", "class": "GaMlp", "param_grid": ga_paras},  # Genetic Algorithm
        {"name": "JADE-MLP", "class": "JadeMlp", "param_grid": jade_paras},  # Differential Evolution
        {"name": "CL-PSO-MLP", "class": "CLPsoMlp", "param_grid": clpso_paras},  # Particle Swarm Optimization
        {"name": "SLO-MLP", "class": "SloMlp", "param_grid": slo_paras},  # Sea Lion Optimization
        {"name": "ISLO-MLP", "class": "IsloMlp", "param_grid": islo_paras},  # Improved Sea Lion Optimization
    ]