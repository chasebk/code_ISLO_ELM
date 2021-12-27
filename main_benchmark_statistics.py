#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 00:30, 01/06/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from pandas import read_csv
from config import Config
from pandas import DataFrame


def save_fast_to_csv(list_results, list_paths, columns):
    for idx, results in enumerate(list_results):
        df = DataFrame(results, columns=columns)
        df.to_csv(list_paths[idx], index=False)
    return True


PROBLEM_SIZE = 30
LIST_FUNCTIONS = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20"]
LIST_MHAS = ["GA", "SAP_DE", "WOA", "COA", "HGS", "LCBO", "CHIO", "SLO", "ImprovedSLO"]
LIST_NAMES = ["GA", "SAP-DE", "HI-WOA", "COA", "HGS", "M-LCO","CHIO", "SLO", "ISLO"]

final_results = []
for func_name in LIST_FUNCTIONS:
    for idx, mha in enumerate(LIST_MHAS):
        filesave = f"{Config.BENCHMARK_RESULTS}/statistics.csv"

        df = read_csv(f"{Config.BENCHMARK_BEST_FIT}/{PROBLEM_SIZE}D_{mha}_best_fit.csv", usecols=["function", "time", "trial", "fit"])
        df_result = df[(df["function"] == func_name)][["time", "fit"]]

        t1 = df_result.min(axis=0).to_numpy().tolist()
        t2 = df_result.mean(axis=0).to_numpy()
        t3 = df_result.max(axis=0).to_numpy().tolist()
        t4 = df_result.std(axis=0).to_numpy()
        t5 = t4 / t2
        t2 = t2.tolist()
        t4 = t4.tolist()

        final = [func_name, LIST_NAMES[idx], t1[0], t3[0], t5[0], t2[0], t4[0], t1[1], t3[1], t5[1], t2[1], t4[1]]
        final_results.append(final)

save_fast_to_csv([final_results], [f"{Config.BM_STATISTICS}"], columns=Config.BM_STT_COLS)

