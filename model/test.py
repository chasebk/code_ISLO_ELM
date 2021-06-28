#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 09:40, 03/06/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from model.SLO import BaseSLO, ImprovedSLO, ISLO
from opfunu.cec_basic.cec2014 import *
from opfunu.cec.cec2014.function import F17 as f11
from opfunu.cec.cec2014.function import F18 as f12
from opfunu.cec.cec2014.function import F20 as f13
from opfunu.cec.cec2015.function import F12 as f18
from numpy import sum

from model.benchmark import f8
# Setting parameters
verbose = True
epoch = 1000
pop_size = 50

# A - Different way to provide lower bound and upper bound. Here are some examples:

## 1. When you have different lower bound and upper bound for each parameters
lb1 = [-100] * 30
ub1 = [100] * 30


def f1(solution):
    return sum(solution ** 2)


from numpy import dot, ones, array, ceil
from opfunu.cec.cec2014.utils import *

SUPPORT_DIMENSION = [2, 10, 20, 30, 50, 100]
SUPPORT_DIMENSION_2 = [10, 20, 30, 50, 100]


# def F1(solution=None, name="Rotated High Conditioned Elliptic Function", shift_data_file="shift_data_1.txt", bias=100):
#     problem_size = len(solution)
#     if problem_size > 100:
#         print("CEC 2014 not support for problem size > 100")
#         return 1
#     if problem_size in SUPPORT_DIMENSION:
#         f_matrix = "M_1_D" + str(problem_size) + ".txt"
#     else:
#         print("CEC 2014 function only support problem size 2, 10, 20, 30, 50, 100")
#         return 1
#     shift_data = load_shift_data__(shift_data_file)[:problem_size]
#     matrix = load_matrix_data__(f_matrix)
#     z = dot(solution - shift_data, matrix)
#     # z = dot(matrix, solution - shift_data)
#     return f1_elliptic__(z) + bias


md1 = ISLO(f8, lb1, ub1, verbose, epoch, pop_size)
best_pos1, best_fit1, list_loss1 = md1.train()
print(md1.solution[1])
