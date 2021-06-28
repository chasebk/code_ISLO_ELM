from cfnn1hl_script import run_cfnn
from gru1hl_script import run_gru
from islo_cfnn_script import run_islo_cfnn
from lstm1hl_script import run_lstm
from mlp1hl_script import run_mlp
from pso_cfnn_script import run_pso_cfnn
from rnn1hl_script import run_rnn
from slno_cfnn_script import run_slno_cfnn

model_names = ['islo-cfnn', 'pso-cfnn', 'slno-cfnn', 'cfnn', 'mlp', 'rnn', 'lstm', 'gru']
run_scripts = [run_islo_cfnn, run_pso_cfnn, run_slno_cfnn, run_cfnn, run_mlp, run_rnn, run_lstm, run_gru]

for i in range(len(model_names)):
    print('start to run {}'.format(model_names[i]))
    run_scripts[i]()