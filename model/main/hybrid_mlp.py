from model.optimizer.swarm import BFO, PSO, SLnO
from model.optimizer.evolutionary import GA, DE, CRO
from model.root.hybrid.root_hybrid_mlp import RootHybridMLP


class GaMLP(RootHybridMLP):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, ga_paras=None):
        RootHybridMLP.__init__(self, root_base_paras, root_hybrid_paras)
        self.ga_paras = ga_paras
        self.filename = "GA_MLP-sliding_{}-nets_{}-ga_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], ga_paras)

    def _training__(self):
        ga = GA.BaseGA(root_algo_paras=self.root_algo_paras, ga_paras = self.ga_paras)
        self.solution, self.loss_train = ga._train__()


class DeMLP(RootHybridMLP):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, de_paras=None):
        RootHybridMLP.__init__(self, root_base_paras, root_hybrid_paras)
        self.de_paras = de_paras
        self.filename = "DE_MLP-sliding_{}-nets_{}-de_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activations"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], de_paras)

    def _training__(self):
        md = DE.BaseDE(root_algo_paras=self.root_algo_paras, de_paras = self.de_paras)
        self.solution, self.loss_train = md._train__()


class PsoMLP(RootHybridMLP):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, pso_paras=None):
        RootHybridMLP.__init__(self, root_base_paras, root_hybrid_paras)
        self.pso_paras = pso_paras
        self.filename = "PSO_MLP-sliding_{}-nets_{}-pso_{}".format(root_base_paras["sliding"],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activations"], root_hybrid_paras["hidden_size"],
             root_hybrid_paras["train_valid_rate"]], pso_paras)

    def _training__(self):
        pso = PSO.BasePSO(root_algo_paras=self.root_algo_paras, pso_paras = self.pso_paras)
        self.solution, self.loss_train = pso._train__()

