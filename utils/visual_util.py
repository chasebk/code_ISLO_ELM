#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 10:17, 06/07/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import platform
from pathlib import Path
import matplotlib.pyplot as plt


def draw_predict(list_lines: list, list_legends: list, xy_labels: list, title: str, filename: str, pathsave:str, exts: list, verbose:True):
    Path(pathsave).mkdir(parents=True, exist_ok=True)
    plt.plot(list_lines[0]) # True
    plt.plot(list_lines[1]) # Predict
    plt.ylabel(xy_labels[1])
    plt.xlabel(xy_labels[0])
    plt.legend(list_legends, loc='upper right')
    plt.title(title)
    for idx, ext in enumerate(exts):
        plt.savefig(f"{pathsave}/{filename}{ext}", bbox_inches='tight')
    if platform.system() != "Linux" and verbose:
        plt.show()
    plt.close()
    return None


def draw_predict_with_error(fig_id=None, data=None, error=None, filename=None, pathsave=None):
    plt.figure(fig_id)
    plt.plot(data[0])
    plt.plot(data[1])
    plt.ylabel('Real value')
    plt.xlabel('Point')
    plt.legend(['Predict y... RMSE= ' + str(error[0]), 'Test y... MAE= ' + str(error[1])], loc='upper right')
    plt.savefig(pathsave + filename + ".png")
    plt.close()
    return None


def draw_raw_time_series_data(data=None, label=None, title=None, filename=None, pathsave=None):
    plt.plot(data)
    plt.xlabel(label["y"])
    plt.ylabel(label["x"])
    plt.title(title, fontsize=8)
    plt.savefig(pathsave + filename + ".pdf")
    plt.close()
    return None


def draw_raw_time_series_data_and_show(data=None, label=None, title=None):
    plt.plot(data)
    plt.xlabel(label["y"])
    plt.ylabel(label["x"])
    plt.title(title, fontsize=8)
    plt.show()
    return None
