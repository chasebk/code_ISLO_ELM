#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 11:19, 03/08/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

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

from model.SLO import BaseSLO as SLO
from model.SLO import ISLO as ISLO
from model.COA import BaseCOA as COA

