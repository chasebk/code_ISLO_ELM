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


    DATA_INPUT = f'{DATA_DIRECTORY}/input_data'
    DATA_RESULTS = f'{DATA_DIRECTORY}/results'
    RESULTS_FOLDER_VISUALIZE = "visualize"
    RESULTS_FOLDER_MODEL = "model"