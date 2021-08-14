#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:47, 07/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, randint, normal, random, choice, rand
from numpy import abs, sign, cos, pi, sin, sqrt, power
from math import gamma
from copy import deepcopy
from mealpy.root import Root


class BaseSLO(Root):
    """
        The original version of: Sea Lion Optimization Algorithm (SLO)
            (Sea Lion Optimization Algorithm)
        Link:
            https://www.researchgate.net/publication/333516932_Sea_Lion_Optimization_Algorithm
            DOI: 10.14569/IJACSA.2019.0100548
        Notes:
            + The original paper is dummy, tons of unclear equations and parameters
            + You can check my question on the ResearchGate link, the authors seem to be scare so they didn't reply.
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            c = 2 - 2 * epoch / self.epoch
            t0 = rand()
            v1 = sin(2 * pi * t0)
            v2 = sin(2 * pi * (1 - t0))
            SP_leader = abs(v1 * (1 + v2) / v2)

            for i in range(self.pop_size):
                if SP_leader < 0.25:
                    if c < 1:
                        pos_new = g_best[self.ID_POS] - c * abs(2*rand()*g_best[self.ID_POS] - pop[i][self.ID_POS])
                    else:
                        ri = choice(list(set(range(0, self.pop_size)) - {i}))       # random index
                        pos_new = pop[ri][self.ID_POS] - c * abs(2 * rand() * pop[ri][self.ID_POS] - pop[i][self.ID_POS])
                else:
                    pos_new = abs(g_best[self.ID_POS] - pop[i][self.ID_POS]) * cos(2*pi*uniform(-1, 1)) + g_best[self.ID_POS]

                pos_new = self.amend_position_random_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                pop[i] = [pos_new, fit]
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class ImprovedSLO(Root):
    """
        My improved version of: Improved Sea Lion Optimization Algorithm (ISLO)
            This version based on Levy-flight
        Link:
            https://www.researchgate.net/publication/333516932_Sea_Lion_Optimization_Algorithm
            DOI: 10.14569/IJACSA.2019.0100548
    """
    ID_LOC_POS = 2
    ID_LOC_FIT = 3

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, c1=2, c2=2, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.c1 = c1
        self.c2 = c2

    def create_solution(self, minmax=0):
        position = uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=position, minmax=minmax)
        local_pos = self.lb + self.ub - position
        local_fit = self.get_fitness_position(position=local_pos, minmax=minmax)
        if fitness < local_fit:
            return [local_pos, local_fit, position, fitness]
        else:
            return [position, fitness, local_pos, local_fit]

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            c = 2 - 2 * epoch / self.epoch
            t0 = rand()
            v1 = sin(2 * pi * t0)                       # -1, 1
            v2 = sin(2 * pi * (1 - t0))                 # -1, 1
            SP_leader = abs(v1 * (1 + v2) / v2)

            for i in range(self.pop_size):
                if SP_leader < 1:
                    if c > 1:  # Exploration
                        dif1 = 2 * rand() * g_best[self.ID_POS] - pop[i][self.ID_POS]
                        dif2 = 2 * rand() * pop[i][self.ID_LOC_POS] - pop[i][self.ID_POS]
                        pos_new = pop[i][self.ID_POS] + c * dif1 + c * dif2
                        # dif1 = abs(g_best[self.ID_POS] - pop[i][self.ID_POS])
                        # dif2 = abs(pop[i][self.ID_LOC_POS] - pop[i][self.ID_POS])
                        # pos_new = pop[i][self.ID_POS] + \
                        #     self.c1 * rand() * (pop[i][self.ID_POS] - c * dif1) + self.c2 * rand() * (pop[i][self.ID_POS] - c * dif2)
                    else:  # Exploitation
                        pos_new = g_best[self.ID_POS] + c * normal(0, 1) * (2 * rand() * g_best[self.ID_POS] - pop[i][self.ID_POS])
                        fit_new = self.get_fitness_position(pos_new)
                        pos_new_oppo = self.lb + self.ub - g_best[self.ID_POS] + rand() * (g_best[self.ID_POS] - pos_new)
                        fit_new_oppo = self.get_fitness_position(pos_new_oppo)
                        if fit_new_oppo < fit_new:
                            pos_new = pos_new_oppo
                else:
                    if rand() < 0.5:        # Exploitation
                        pos_new = g_best[self.ID_POS] + cos(2 * pi * uniform(-1, 1)) * abs(g_best[self.ID_POS] - pop[i][self.ID_POS])
                    else:                   # Exploration
                        pos_new = self.levy_flight(epoch=epoch, position=pop[i][self.ID_POS], g_best_position=g_best[self.ID_POS])
                        # pos_new = pop[i][self.ID_POS] + self.step_size_by_levy_flight(0.01, 1.5, 2) * (g_best[self.ID_POS] - pop[i][self.ID_POS])

                pos_new = self.amend_position_random_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_LOC_FIT]:
                    pop[i] = [pos_new, fit, deepcopy(pos_new), deepcopy(fit)]
                else:
                    pop[i][self.ID_POS] = pos_new
                    pop[i][self.ID_FIT] = fit
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class ISLO(Root):
    """
        My improved version of: Improved Sea Lion Optimization Algorithm (ISLO)
            (Sea Lion Optimization Algorithm)
        Link:
            https://www.researchgate.net/publication/333516932_Sea_Lion_Optimization_Algorithm
            DOI: 10.14569/IJACSA.2019.0100548
    """
    ID_POS_LOC = 2
    ID_POS_FIT = 3

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, c1=1.2, c2=1.2, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.c1 = c1
        self.c2 = c2

    def create_solution(self, minmax=0):
        position = uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=position, minmax=minmax)
        local_pos = self.lb + self.ub - position
        local_fit = self.get_fitness_position(position=local_pos, minmax=minmax)
        if fitness < local_fit:
            return [local_pos, local_fit, position, fitness]
        else:
            return [position, fitness, local_pos, local_fit]

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            c = 2 - 2 * epoch / self.epoch
            t0 = rand()
            v1 = sin(2 * pi * t0)
            v2 = sin(2 * pi * (1 - t0))
            SP_leader = abs(v1 * (1 + v2) / v2)

            for i in range(self.pop_size):
                if SP_leader < 0.5:
                    if c < 1:   # Exploitation
                        # pos_new = g_best[self.ID_POS] - c * abs(2 * rand() * g_best[self.ID_POS] - pop[i][self.ID_POS])
                        dif1 = abs(2 * rand() * g_best[self.ID_POS] - pop[i][self.ID_POS])
                        dif2 = abs(2 * rand() * pop[i][self.ID_POS_LOC] - pop[i][self.ID_POS])
                        pos_new = self.c1 * rand() * (pop[i][self.ID_POS] - c * dif1) + self.c2*rand() *(pop[i][self.ID_POS] - c*dif2)
                    else:   # Exploration
                        # ri = choice(list(set(range(0, self.pop_size)) - {i}))  # random index
                        # pos_new = pop[ri][self.ID_POS] - c * abs(2 * rand() * pop[ri][self.ID_POS] - pop[i][self.ID_POS])
                        pos_new = g_best[self.ID_POS] + c * normal(0, 1, self.problem_size) * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                        fit_new = self.get_fitness_position(pos_new)
                        pos_new_oppo = self.lb + self.ub - g_best[self.ID_POS] + rand() * (g_best[self.ID_POS] - pos_new)
                        fit_new_oppo = self.get_fitness_position(pos_new_oppo)
                        if fit_new_oppo < fit_new:
                            pos_new = pos_new_oppo
                # elif 0.5 <= SP_leader <= 1:
                #     # pos_new = self.levy_flight(epoch=epoch, position=pop[i][self.ID_POS], g_best_position=g_best[self.ID_POS])
                #     pos_new = pop[i][self.ID_POS] + self.step_size_by_levy_flight(case=3)
                else:
                    pos_new = abs(g_best[self.ID_POS] - pop[i][self.ID_POS]) * cos(2 * pi * uniform(-1, 1)) + g_best[self.ID_POS]
                pos_new = self.amend_position_random_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_POS_FIT]:
                    pop[i] = [pos_new, fit, deepcopy(pos_new), deepcopy(fit)]
                else:
                    pop[i][self.ID_POS] = pos_new
                    pop[i][self.ID_FIT] = fit
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

