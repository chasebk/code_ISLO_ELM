#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 13:57, 08/07/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from mealpy.evolutionary_based import GA, DE, FPA
from mealpy.swarm_based import ABC, PSO, HHO, GWO, WOA, SpaSA, MFO, ALO, GOA, SalpSO, DO, FA, BeesA, ACOR, NMRA
from mealpy.swarm_based import FireflyA, BA, FOA, SSO, SSA, EHO, JA, BSA, SHO, SRSR, MSA, BES, PFA, SFO, MRFO, HGS
from mealpy.bio_based import BBO, IWO, SMA, EOA, SBO, VCS
from mealpy.human_based import TLO, LCBO, ICA, CA, BRO, BSO, CHIO, FBIO, GSKA, QSA, SARO, SSDO
from mealpy.physics_based import MVO, EO, SA, HGSO, ASO, EFO, NRO, TWO, WDO
from mealpy.math_based import SCA, HC
from mealpy.system_based import WCA, AEO, GCO
from mealpy.dummy import BOA
from mealpy.swarm_based import SLO
from model.app.hybrid_mlp import HybridMlp


## Evolutionary Group

class GaMlp(HybridMlp):
    def __init__(self, mha_paras=None):
        super().__init__()
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.pc = mha_paras["pc"]
        self.pm = mha_paras["pm"]
        self.filename = f"{self.epoch}-{self.pop_size}-{self.pc}-{self.pm}"

    # fit hybrid MLP network to training data
    def fit_model(self):
        self.opt = GA.BaseGA(obj_func=self.objective_function, lb=self.lb, ub=self.ub, verbose=self.verbose,
                       epoch=self.epoch, pop_size=self.pop_size, pc=self.pc, pm=self.pm)
        self.solution, self.fitness, self.list_loss = self.opt.train()


class JadeMlp(HybridMlp):
    def __init__(self, mha_paras=None):
        super().__init__()
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.miu_f = mha_paras["miu_f"]
        self.miu_cr = mha_paras["miu_cr"]
        self.pp = mha_paras["pp"]
        self.cc = mha_paras["cc"]
        self.filename = f"{self.epoch}-{self.pop_size}-{self.miu_f}-{self.miu_cr}-{self.pp}-{self.cc}"

    # fit hybrid MLP network to training data
    def fit_model(self):
        self.opt = DE.JADE(obj_func=self.objective_function, lb=self.lb, ub=self.ub, verbose=self.verbose,
                             epoch=self.epoch, pop_size=self.pop_size, miu_f=self.miu_f, miu_cr=self.miu_cr,
                           p=self.pp, c=self.cc)
        self.solution, self.fitness, self.list_loss = self.opt.train()


## Swarm Group

class CLPsoMlp(HybridMlp):
    def __init__(self, mha_paras=None):
        super().__init__()
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.c_local = mha_paras["c_local"]
        self.w_min = mha_paras["w_min"]
        self.w_max = mha_paras["w_max"]
        self.filename = f"{self.epoch}-{self.pop_size}-{self.c_local}-{self.w_min}-{self.w_max}"

    def fit_model(self):
        self.opt = PSO.CL_PSO(self.objective_function, self.lb, self.ub, self.verbose,
                               self.epoch, self.pop_size, self.c_local, self.w_min, self.w_max)
        self.solution, self.fitness, self.list_loss = self.opt.train()


class SloMlp(HybridMlp):
    def __init__(self, mha_paras=None):
        super().__init__()
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.epoch}-{self.pop_size}"

    def fit_model(self):
        self.opt = SLO.BaseSLO(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size)
        self.solution, self.fitness, self.list_loss = self.opt.train()


class IsloMlp(HybridMlp):
    def __init__(self, mha_paras=None):
        super().__init__()
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.epoch}-{self.pop_size}"

    def fit_model(self):
        self.opt = SLO.ISLO(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size)
        self.solution, self.fitness, self.list_loss = self.opt.train()


class FpaMlp(HybridMlp):
    def __init__(self, mha_paras=None):
        super().__init__()
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.epoch}-{self.pop_size}"

    def fit_model(self):
        self.opt = FPA.BaseFPA(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size)
        self.solution, self.fitness, self.list_loss = self.opt.train()


class HhoMlp(HybridMlp):
    def __init__(self, mha_paras=None):
        super().__init__()
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.epoch}-{self.pop_size}"

    def fit_model(self):
        self.opt = HHO.BaseHHO(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size)
        self.solution, self.fitness, self.list_loss = self.opt.train()


class HgsMlp(HybridMlp):
    def __init__(self, mha_paras=None):
        super().__init__()
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.epoch}-{self.pop_size}"

    def fit_model(self):
        self.opt = HGS.OriginalHGS(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size)
        self.solution, self.fitness, self.list_loss = self.opt.train()


class NroMlp(HybridMlp):
    def __init__(self, mha_paras=None):
        super().__init__()
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.epoch}-{self.pop_size}"

    def fit_model(self):
        self.opt = NRO.BaseNRO(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size)
        self.solution, self.fitness, self.list_loss = self.opt.train()


class TloMlp(HybridMlp):
    def __init__(self, mha_paras=None):
        super().__init__()
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.epoch}-{self.pop_size}"

    def fit_model(self):
        self.opt = TLO.BaseTLO(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size)
        self.solution, self.fitness, self.list_loss = self.opt.train()


class FbioMlp(HybridMlp):
    def __init__(self, mha_paras=None):
        super().__init__()
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.epoch}-{self.pop_size}"

    def fit_model(self):
        self.opt = FBIO.BaseFBIO(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size)
        self.solution, self.fitness, self.list_loss = self.opt.train()


class SmaMlp(HybridMlp):
    def __init__(self, mha_paras=None):
        super().__init__()
        self.epoch = mha_paras["epoch"]
        self.pop_size = mha_paras["pop_size"]
        self.filename = f"{self.epoch}-{self.pop_size}"

    def fit_model(self):
        self.opt = SMA.BaseSMA(self.objective_function, self.lb, self.ub, self.verbose, self.epoch, self.pop_size)
        self.solution, self.fitness, self.list_loss = self.opt.train()