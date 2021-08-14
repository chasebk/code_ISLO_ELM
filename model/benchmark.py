#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 21:45, 31/05/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy import sum, cos, pi

from opfunu.type_based.multi_modal import Functions as func_multi
from opfunu.type_based.uni_modal import Functions as func_uni

from opfunu.cec.cec2014.function import F17 as f11
from opfunu.cec.cec2014.function import F18 as f12
from opfunu.cec.cec2014.function import F20 as f13
from opfunu.cec.cec2014.function import F6 as f14
from opfunu.cec.cec2014.function import F8 as f15

from opfunu.cec.cec2015.function import F9 as f16
from opfunu.cec.cec2015.function import F10 as f17
from opfunu.cec.cec2015.function import F12 as f18
from opfunu.cec.cec2015.function import F14 as f19
from opfunu.cec.cec2015.function import F15 as f20

f1 = func_uni()._sum_squres__
f2 = func_uni()._zakharov__
f3 = func_uni()._schwefel_2_22__
f4 = func_uni()._dixon_price__
f5 = func_uni()._rosenbrock__

f6 = func_multi()._ackley__
f7 = func_multi()._griewank__
# f8 = func_multi()._shubert__
f9 = func_multi()._schaffer_f6__
f10 = func_multi()._salomon__

from mealpy.evolutionary_based.GA import BaseGA as GA
from mealpy.evolutionary_based.DE import SADE as SADE
from mealpy.evolutionary_based.DE import SAP_DE as SAP_DE
from mealpy.evolutionary_based.DE import SHADE as SHADE
from mealpy.evolutionary_based.DE import L_SHADE as L_SHADE

from mealpy.swarm_based.WOA import BaseWOA as WOA
from mealpy.swarm_based.WOA import HI_WOA as HI_WOA
from mealpy.swarm_based.HGS import OriginalHGS as HGS
# from mealpy.swarm_based.COA import BaseCOA as COA

from mealpy.human_based.LCBO import BaseLCBO as LCBO
from mealpy.human_based.CHIO import BaseCHIO as CHIO

from mealpy.physics_based.TWO import OppoTWO as OTWO
from mealpy.physics_based.HGSO import OppoHGSO as OBL_HGSO
# Opposition based Henry gas solubility optimization as a novel algorithm for PID control of DC motor

from model.SLO import BaseSLO as SLO
from model.SLO import ISLO as ISLO
from model.COA import BaseCOA as COA
from model.SLO import ImprovedSLO


def f8(solution):
    return 10 * len(solution) + sum(solution**2 - 10*cos(2*pi*solution))
