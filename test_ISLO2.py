#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 18:49, 14/08/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy import sum
from model.SLO import ImprovedSLO as ImprovedSLO2

TRIALS = 1
PROBLEM_SIZE = 30
LB = [-100] * PROBLEM_SIZE
UB = [100] * PROBLEM_SIZE
EPOCH = 1000
POP_SIZE = 50


def objective_function(solution):
    return sum(solution ** 2)

md = ImprovedSLO2(objective_function, LB, UB, True, EPOCH, POP_SIZE)
_, best_fit, list_loss = md.train()

