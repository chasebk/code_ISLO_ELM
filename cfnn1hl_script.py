from sklearn.model_selection import ParameterGrid
from model.main.traditional_ffnn import CFNN1HL
from utils.IOUtil import read_dataset_file
from utils.SettingPaper import cfnn1hl_paras_final as param_grid
from utils.SettingPaper import ggtrace_cpu, ggtrace_ram, ggtrace_multi_cpu, ggtrace_multi_ram, traffic_eu, traffic_uk, worldcup
from datetime import datetime

rv_data = [ggtrace_cpu, ggtrace_ram, traffic_eu, traffic_uk]
data_file = ["gg_trace_5m", "gg_trace_5m", "it_eu_5m", "it_uk_5m"]
test_type = "stability"                ### normal: for normal test, stability: for n_times test
run_times = None

if test_type == "normal":           ### For normal test
    run_times = 1
    pathsave = "paper/results/test/"
    all_model_file_name = "nn_log_models"
elif test_type == "stability":      ### For stability test (n times run with the same parameters)
    run_times = 1
    pathsave = "paper/results/stability/"
    all_model_file_name = "stability_cfnn1hl"
else:
    pass

def train_model(item, dataset, requirement_variables):
    root_base_paras = {
        "dataset": dataset,
        "data_idx": (0.8, 0, 0.2),
        "sliding": item["sliding"],
        "multi_output": requirement_variables[2],
        "output_idx": requirement_variables[3],
        "method_statistic": 0,  # 0: sliding window, 1: mean, 2: min-mean-max, 3: min-median-max
        "log_filename": all_model_file_name,
        "path_save_result": pathsave + requirement_variables[4],
        "test_type": test_type,
        "draw": True,
        "print_train": 0 # 0: nothing, 1 : full detail, 2: short version
    }
    root_mlnn_paras = {
        "hidden_sizes": item["hidden_sizes"], "epoch": item["epoch"], "batch_size": item["batch_size"],
        "learning_rate": item["learning_rate"], "activations": item["activations"],
        "optimizer": item["optimizer"], "loss": item["loss"]
    }
    md = CFNN1HL(root_base_paras=root_base_paras, root_mlnn_paras=root_mlnn_paras)
    md._running__()


def run_cfnn():
    for _ in range(run_times):
        for loop in range(len(rv_data)):
            requirement_variables = rv_data[loop]
            filename = requirement_variables[0] + data_file[loop] + ".csv"
            dataset = read_dataset_file(filename, requirement_variables[1])
            # Create combination of params.
            for item in list(ParameterGrid(param_grid)):
                start_time = datetime.now()
                train_model(item, dataset, requirement_variables)
                time = datetime.now() - start_time
                print("running cfnn with data {} and time {}".format(loop, time.total_seconds()))