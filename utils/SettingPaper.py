###### Config for test

###: Variables
traffic_eu = [
    "data/formatted/",    # pd.readfilepath
    [4],                    # usecols trong pd
    False,                  # multi_output
    None,                   # output_idx
    "eu/",                  # path_save_result
]
traffic_uk = [
    "data/formatted/",    # pd.readfilepath
    [2],                    # usecols trong pd
    False,                  # multi_output
    None,                   # output_idx
    "uk/",                  # path_save_result
]

worldcup = [
    "data/formatted/",    # pd.readfilepath
    [2],            # usecols trong pd
    False,          # multi_output
    None,           # output_idx
    "wc/",    # path_save_result
]

ggtrace_cpu = [
    "data/formatted/",    # pd.readfilepath
    [1],         # usecols trong pd
    False,          # multi_output
    None,              # output_idx
    "cpu/",     # path_save_result
]

ggtrace_ram = [
    "data/formatted/",    # pd.readfilepath
    [2],             # usecols trong pd
    False,              # multi_output
    None,                  # output_idx
    "ram/",       # path_save_result
]

ggtrace_multi_cpu = [
    "data/formatted/",    # pd.readfilepath
    [1, 2],         # usecols trong pd
    False,          # multi_output
    0,              # output_idx
    "multi_cpu/",     # path_save_result
]

ggtrace_multi_ram = [
    "data/formatted/",    # pd.readfilepath
    [1, 2],             # usecols trong pd
    False,              # multi_output
    1,                  # output_idx
    "multi_ram/",       # path_save_result
]


######################## Paras according to the paper

####: CFNN-1HL
cfnn1hl_paras_final = {
    "sliding": [3],
    "hidden_sizes" : [[10]],
    "activations": [("sigmoid", "sigmoid")],  # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "learning_rate": [0.0001],
    "epoch": [1000],
    "batch_size": [64],
    "optimizer": ["adam"],   # GradientDescentOptimizer, AdamOptimizer, AdagradOptimizer, AdadeltaOptimizer
    "loss": ["mse"]
}

####: FFNN-1HL
ffnn1hl_paras_final = {
    "sliding": [3],
    "hidden_sizes" : [[10]],
    "activations": [("tanh", "tanh")],  # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "learning_rate": [0.0001],
    "epoch": [1000],
    "batch_size": [64],
    "optimizer": ["adam"],   # GradientDescentOptimizer, AdamOptimizer, AdagradOptimizer, AdadeltaOptimizer
    "loss": ["mse"]
}

####: RNN-1HL
rnn1hl_paras_final = {
    "sliding": [3],
    "hidden_sizes" : [[10]],
    "activations": [("elu", "elu")],  # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "learning_rate": [0.0001],
    "epoch": [1000],
    "batch_size": [64],
    "optimizer": ["adam"],   # GradientDescentOptimizer, AdamOptimizer, AdagradOptimizer, AdadeltaOptimizer
    "loss": ["mse"],
    "dropouts": [[0.0]]
}

####: LSTM-1HL
lstm1hl_paras_final = {
    "sliding": [3],
    "hidden_sizes" : [[10]],
    "activations": [("elu", "elu"), ('sigmoid', 'sigmoid')],  # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "learning_rate": [0.01],
    "epoch": [1000],
    "batch_size": [32],
    "optimizer": ["adam"],   # GradientDescentOptimizer, AdamOptimizer, AdagradOptimizer, AdadeltaOptimizer
    "loss": ["mse"],
    "dropouts": [[0.1]]
}

####: GRU-1HL
gru1hl_paras_final = {
    "sliding": [3],
    "hidden_sizes" : [[10]],
    "activations": [("elu", "elu")],  # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "learning_rate": [0.0001],
    "epoch": [1000],
    "batch_size": [64],
    "optimizer": ["adam"],   # GradientDescentOptimizer, AdamOptimizer, AdagradOptimizer, AdadeltaOptimizer
    "loss": ["mse"],
    "dropouts": [[0.0]]
}

#### : SLnO-CFNN
slno_cfnn_paras_final = {
    "sliding": [3],
    "hidden_size" : [6],
    "activation": [3],             # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(1, 0)],

    "epoch": [1000],
    "pop_size": [200],                  # 100 -> 900
    "sp_leader": [0.25],
    "domain_range": [(-1, 1)]           # lower and upper bound
}


#### : GA-CFNN
ga_cfnn_paras_final = {
    "sliding": [3],
    "hidden_size" : [5],
    "activation": [3],             # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(1, 0)],

    "epoch": [1000],
    "pop_size": [200],                  # 100 -> 900
    "pc": [0.95],                       # 0.85 -> 0.97
    "pm": [0.025],                      # 0.005 -> 0.10
    "domain_range": [(-1, 1)]           # lower and upper bound
}


#### : PSO-CFNN
pso_cfnn_paras_final = {
    "sliding": [3],
    "hidden_size" : [6],
    "activation": [3],             # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(1, 0)],

    "epoch": [1000],
    "pop_size": [200],                  # 100 -> 900
    "w_minmax": [(0.4, 0.9)],  # [0-1] -> [0.4-0.9]      Trong luong cua con chim
    "c_minmax": [(1.6, 0.6)],  # [(1.2, 1.2), (0.8, 2.0), (1.6, 0.6)]     # [0-2]   Muc do anh huong cua local va global
    # r1, r2 : random theo tung vong lap
    # delta(t) = 1 (do do: x(sau) = x(truoc) + van_toc
    "domain_range": [(-1, 1)]           # lower and upper bound
}


#### : WOA-CFNN
woa_cfnn_paras_final = {
    "sliding": [3],
    "hidden_size" : [5],
    "activation": [3],             # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(1, 0)],

    "epoch": [1000],
    "pop_size": [200],                  # 100 -> 900
    "domain_range": [(-1, 1)]           # lower and upper bound
}


#### : ISLO-CFNN
islo_cfnn_paras_final = {
    "sliding": [2],
    "hidden_size": [5],
    "activation": [3],             # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(1, 0)],

    "epoch": [1000],
    "pop_size": [200],                  # 100 -> 900
    "sp_leader": [0.25],
    "domain_range": [(-1, 1)]           # lower and upper bound
}