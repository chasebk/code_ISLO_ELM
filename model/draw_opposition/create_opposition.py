#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 14:49, 07/05/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy import array, sum, ones, clip
from numpy.random import uniform, normal, rand
import matplotlib.pyplot as plt


def objective_function(solution):
    return space + sum(solution**2)

def amend_solution(sol, lb, ub):
    return clip(sol, lb, ub)

def create_solution(lb, ub):
    x = uniform(lb, ub)
    fit = objective_function(x)
    return [x, fit]

def create_opposition_solution(x, x_best, C, lb, ub):
    x1 = x_best + normal() * C * (x_best - x)
    x_oppo = lb + ub - x_best + rand() * (x_best - x1)
    x_oppo = amend_solution(x_oppo, lb, ub)
    fit = objective_function(x_oppo)
    return [x_oppo, fit]


space = 2

lb2 = array([-5, -10])
ub2 = array([5, 10])

lb3 = array([-5, -10, -15])
ub3 = array([5, 10, 15])

g_max = 1

x_best = array([-1.4, 0.5])
fit_best = objective_function(x_best)

list_solutions = []
list_opposition_solutions = []
list_local_best = []
for g in range(g_max):
    C = 2 * (1 - g / g_max)
    local_best = x_best + normal(0, 1, len(lb2))
    sol2 = create_solution(lb2, ub2)
    sol2_oppo = create_opposition_solution(sol2[0], local_best, C, lb2, ub2)
    print(f"x: {sol2[0]}, x_oppo: {sol2_oppo[0]}")

    list_solutions.append(sol2[0])
    list_opposition_solutions.append(sol2_oppo[0])
    list_local_best.append(local_best)

list_solutions = array(list_solutions)
list_opposition_solutions = array(list_opposition_solutions)
list_local_best = array(list_local_best)

# Create data
g1 = (list_solutions[:, 0].flatten(), list_solutions[:, 1].flatten())
g2 = (list_opposition_solutions[:, 0].flatten(), list_opposition_solutions[:, 1].flatten())
g3 = (list_local_best[:, 0].flatten(), list_local_best[:, 1].flatten())

data = (g1, g2, g3)
colors = ("blue", "green", "darkorange")
groups = ("x", "x_opposition", "local_best")
markers = ("o", "s", "v")

# Create plot
fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
ax = plt.axes(facecolor='whitesmoke')
for data, color, group, marker in zip(data, colors, groups, markers):
    x, y = data
    ax.scatter(x, y, alpha=0.8, marker=marker, c=color, edgecolors='none', s=30, label=group)

plt.plot(0, 0, 'r*', label='Optimal')

plt.xlim([lb2[0], ub2[0]])
plt.ylim(lb2[1], ub2[1])

plt.title(f'x and x_opposition after {g_max} generations')
# plt.legend(loc=2)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.savefig(f"{g_max}-generations.pdf", bbox_inches='tight')
plt.show()



#
#
# import matplotlib
# matplotlib.use('TkAgg')
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
#
# xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# ys = [5, 6, 2, 3, 13, 4, 1, 2, 4, 8]
# zs = [2, 3, 3, 3, 5, 7, 9, 11, 9, 10]
#
# xt = [-1, -2, -3, -4, -5, -6, -7, 8, -9, -10]
# yt = [-5, -6, -2, -3, -13, -4, -1, 2, -4, -8]
# zt = [-2, -3, -3, -3, -5, -7, 9, -11, -9, -10]
#
# ax.scatter(xs, ys, zs, c='r', marker='o')
# ax.scatter(xt, yt, zt, c='b', marker='^')
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# plt.show()