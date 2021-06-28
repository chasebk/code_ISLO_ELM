#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 09:06, 17/06/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
from numpy import where, column_stack


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


def make_timeseries(data, sequence_len=cfg_sequence_len, sequence_len_y=cfg_sequence_len_y, steps_ahead=cfg_steps_ahead):
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
