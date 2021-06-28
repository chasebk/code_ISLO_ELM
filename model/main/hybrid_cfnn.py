from model.optimizer.swarm import PSO, SLnO, WOA
from model.optimizer.evolutionary import GA, DE
from model.root.hybrid.root_hybrid_cfnn import RootHybridCFNN


class GaCFNN(RootHybridCFNN):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, ga_paras=None):
        RootHybridCFNN.__init__(self, root_base_paras, root_hybrid_paras)
        self.ga_paras = ga_paras
        self.filename = "GA_CFNN-sliding_{}-nets_{}-ga_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], ga_paras)

    def _training__(self):
        ga = GA.BaseGA(root_algo_paras=self.root_algo_paras, ga_paras = self.ga_paras)
        self.solution, self.loss_train = ga._train__()


class DeCFNN(RootHybridCFNN):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, de_paras=None):
        RootHybridCFNN.__init__(self, root_base_paras, root_hybrid_paras)
        self.de_paras = de_paras
        self.filename = "DE_CFNN-sliding_{}-nets_{}-de_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activations"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], de_paras)

    def _training__(self):
        md = DE.BaseDE(root_algo_paras=self.root_algo_paras, de_paras = self.de_paras)
        self.solution, self.loss_train = md._train__()


class PsoCFNN(RootHybridCFNN):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, pso_paras=None):
        RootHybridCFNN.__init__(self, root_base_paras, root_hybrid_paras)
        self.pso_paras = pso_paras
        self.filename = "PSO_CFNN-sliding_{}-nets_{}-pso_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], pso_paras)

    def _training__(self):
        pso = PSO.BasePSO(root_algo_paras=self.root_algo_paras, pso_paras = self.pso_paras)
        self.solution, self.loss_train = pso._train__()


class WoaCFNN(RootHybridCFNN):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, woa_paras=None):
        RootHybridCFNN.__init__(self, root_base_paras, root_hybrid_paras)
        self.woa_paras = woa_paras
        self.filename = "WOA_CFNN-sliding_{}-nets_{}-slno_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], woa_paras)

    def _training__(self):
        woa = WOA.BaseWOA(root_algo_paras=self.root_algo_paras, woa_paras = self.woa_paras)
        self.solution, self.loss_train = woa._train__()


class SlnoCFNN(RootHybridCFNN):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, slno_paras=None):
        RootHybridCFNN.__init__(self, root_base_paras, root_hybrid_paras)
        self.slno_paras = slno_paras
        self.filename = "SLnO_CFNN-sliding_{}-nets_{}-slno_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], slno_paras)

    def _training__(self):
        slno = SLnO.BaseSLnO(root_algo_paras=self.root_algo_paras, slno_paras = self.slno_paras)
        self.solution, self.loss_train = slno._train__()


class IsloCFNN(RootHybridCFNN):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, slno_paras=None):
        RootHybridCFNN.__init__(self, root_base_paras, root_hybrid_paras)
        self.slno_paras = slno_paras
        self.filename = "ISLO_CFNN-sliding_{}-nets_{}-islo_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], slno_paras)

    def _training__(self):
        islo = SLnO.ISLO(root_algo_paras=self.root_algo_paras, slno_paras = self.slno_paras)
        self.solution, self.loss_train = islo._train__()


