#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 14:35, 28/06/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from pandas import read_csv
from numpy import dot, where, column_stack, mean, ndarray, abs, isnan, nan, sqrt
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set(context='paper', style='whitegrid', color_codes=True)
sns.set_palette(sns.color_palette(["#017b92", "#f97306", "#0485d1"]))  # ["jade green", "orange", "blue"]

cfg_sequence_len = 36  # default=24
cfg_sequence_len_y = 1
cfg_steps_ahead = 1

cfg_batch_size = 128
cfg_units = 6  # default=36
cfg_dropout_rate = 0.2

cfg_num_epochs = 1000
cfg_epochs_patience = 10

cfg_fig_size_x = 20
cfg_fig_size_y = 5

data_train_filename = 'data/formatted/gg_5m_train.tsv'
data_test_filename = 'data/formatted/gg_5m_test.tsv'

model_name = 'MHA_model'


# pandas dataframe to numpy array
def read_data(filename):
    df = read_csv(filename, sep='\t', skiprows=0, skipfooter=0, engine='python')
    data = df.values
    print('read_data: ', filename, '\t', data.shape[1], data.dtype, '\n', list(df))
    return data


# data is numpy array
def transform(data, epsilon=1):
    data = where(data < 0, epsilon, data)
    return data


# Scale all metrics but each separately: normalization or standardization
def normalize(data, scaler=None):
    if not scaler:
        scaler = MinMaxScaler(feature_range=(0, 1))
        norm_data = scaler.fit_transform(data)
    else:
        norm_data = scaler.transform(data)
    # print('\nnormalize:', norm_data.shape)
    return norm_data, scaler


def make_timeseries(data,
                    sequence_len=cfg_sequence_len,
                    sequence_len_y=cfg_sequence_len_y,
                    steps_ahead=cfg_steps_ahead
                    ):
    data_x = data_y = data

    if sequence_len_y > 1:
        for i in range(1, sequence_len_y):
            data_y = column_stack((data_y[:-1], data[i:]))
        data_x = data_x[:-(sequence_len_y - 1)]

    if steps_ahead > 1:
        data_x = data_x[:-(steps_ahead - 1)]
        data_y = data_y[steps_ahead - 1:]

    tsg_data = TimeseriesGenerator(data_x, data_y, length=sequence_len,
                                   sampling_rate=1, stride=1, batch_size=cfg_batch_size)
    # x, y = tsg_data[0]
    # print('\ttsg x.shape=', x.shape, '\n\tx=', x, '\n\ttsg y.shape=', y.shape, '\n\ty=', y)
    return tsg_data


def transform_invert(data, denorm, sequence_len=cfg_sequence_len, steps_ahead=cfg_steps_ahead):
    begin = sequence_len + steps_ahead - 1  # indexing is from 0
    end = begin + len(denorm)
    Y = data[begin:end]  # excludes the end index
    return denorm, Y


def fit_model(data_train, data_test, model, epochs, scaler, callbacks_list):
    trans_train = transform(data_train)
    norm_train, _ = normalize(trans_train, scaler)
    tsg_train = make_timeseries(norm_train)

    trans_test = transform(data_test)
    norm_test, _ = normalize(trans_test, scaler)
    tsg_test = make_timeseries(norm_test)
    history = model.fit(tsg_train, epochs=epochs, callbacks=callbacks_list, validation_data=tsg_test)
    return model, history


def predict(data_test, model, scaler):
    trans_test = transform(data_test)
    norm_test, _ = normalize(trans_test, scaler)
    tsg_test = make_timeseries(norm_test)
    return model.predict(tsg_test)


def mape(y_true, y_pred):
    assert isinstance(y_true, ndarray), 'numpy array expected for y_true in mape'
    assert isinstance(y_pred, ndarray), 'numpy array expected for y_pred in mape'
    score = []
    for i in range(y_true.shape[1]):
        try:
            s = mean(abs((y_true[:, i] - y_pred[:, i]) / y_true[:, i])) * 100
            if isnan(s):
                s = str(s)
            score.append(s)
        except ZeroDivisionError:
            score.append(str(nan))
    return score


# @giang: RMSE for numpy array
def rmse(a, b):
    score = []
    for i in range(a.shape[1]):
        score.append(sqrt(mean_squared_error(a[:, i], b[:, i])))
    return score


# @giang: cosine similarity for two numpy arrays, <-1.0, 1.0>
def cosine(a, b):
    score = []
    for i in range(a.shape[1]):
        cos_sim = dot(a[:, i], b[:, i]) / (norm(a[:, i]) * norm(b[:, i]))
        score.append(cos_sim)
    return score


# @giang: R^2 (coefficient of determination) regression score, <-1.0, 1.0>, not a symmetric function
def r2(a, b):
    score = []
    for i in range(a.shape[1]):
        score.append(r2_score(a[:, i], b[:, i]))
    return score


def eval_predictions(pred_test, Y_test, model_type):
    print('\nEvaluation with real values - One step')
    results = [model_type]

    err_train = err_test = 0
    for m in ['MAPE', 'RMSE', 'R2', 'COSINE']:
        if m == 'MAPE':
            err_test = mape(Y_test, pred_test)
        elif m == 'RMSE':
            err_test = rmse(Y_test, pred_test)
        elif m == 'R2':
            err_test = r2(Y_test, pred_test)
        elif m == 'COSINE':
            err_test = cosine(Y_test, pred_test)
        results.append([m, err_train, err_test])

    line = results[0]  # model_type
    for r in results[1:]:
        line += '\t' + r[0] + '\t'  # SMAPE, MAPE, R2, COSINE
        line += '\t'.join(x if isinstance(x, str) else str("{0:0.4f}".format(x)) for x in r[2])  # test
    print(line)
    return line


def plot_predictions(pred_test, Y_test, multivariate,
                     fig_x=cfg_fig_size_x,
                     fig_y=cfg_fig_size_y
                     ):
    plt.rcParams["figure.figsize"] = (fig_x, fig_y)
    if multivariate > 1:
        fig, ax = plt.subplots(multivariate, sharex=False, figsize=(fig_x, multivariate * fig_y))
        for i in range(multivariate):
            ax[i].plot(Y_test[:, i])
            ax[i].plot(pred_test[:, i])
    else:
        fig, ax = plt.subplots(figsize=(fig_x, multivariate * fig_y))
        ax.plot(Y_test[:, 0])
        ax.plot(pred_test[:, 0])

    fig.tight_layout()
    plt.savefig('plot_image', bbox_inches='tight')
    plt.show()
    return


data_train = read_data(data_train_filename)
trans_train = transform(data_train)
norm_train, scaler = normalize(trans_train)

# save scaler
scaler_filename = model_name + '.scaler'
joblib.dump(scaler, scaler_filename)
print('Scaler saved to: ', scaler_filename)

data_test = read_data(data_test_filename)

# create model
print('Model typ: MLP')
multivariate = data_train.shape[1]

x = Input(shape=(cfg_sequence_len, multivariate))
# h = Dense(units=cfg_units,
#           activation='elu',
#           kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#           bias_regularizer=regularizers.l2(1e-4),
#           activity_regularizer=regularizers.l2(1e-5))(x)
h = Dense(units=cfg_units, activation='elu')(x)
h = Flatten()(h)
y = Dense(units=multivariate * cfg_sequence_len_y, activation='sigmoid')(h)

model = Model(inputs=x, outputs=y)
print(model.summary())

# compile model
opt = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse', 'mae'])  # 'cosine', 'mape'

earlystops = EarlyStopping(monitor='loss', patience=cfg_epochs_patience, verbose=1)
callbacks_list = [earlystops]

# fit model
model, history = fit_model(data_train, data_test, model, 1000, scaler, callbacks_list)

# save model
model.save(model_name)
print('\nSave trained model: ', model_name)

# plot
# print(history.history.keys())
plt.rcParams["figure.figsize"] = (12, 8)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

pred_model = predict(data_test, model, scaler)

eval_line = ''
for i in range(cfg_sequence_len_y):
    one_y_test = pred_model[:, i * multivariate:(i + 1) * multivariate]
    denorm_test = scaler.inverse_transform(one_y_test)
    pred_test, Y_test = transform_invert(data_test, denorm_test, cfg_sequence_len, cfg_steps_ahead)

    # Evaluate with real values
    eval_line += str(i + 1) + '\t' + eval_predictions(pred_test, Y_test, 'NN') + '\n'

    # Plot
    plot_predictions(pred_test[:50], Y_test[:50], multivariate)

