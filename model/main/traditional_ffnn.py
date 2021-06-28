from keras.models import Sequential
from keras.layers import Dense, Add
import keras
from keras import backend as K
import tensorflow as tf
from model.root.traditional.root_mlnn import RootMlnn

class CFNN1HL(RootMlnn):
    def __init__(self, root_base_paras=None, root_mlnn_paras=None):
        RootMlnn.__init__(self, root_base_paras, root_mlnn_paras)
        self.filename = "CFNN-1H-sliding_{}-net_para_{}".format(root_base_paras["sliding"], root_mlnn_paras)

    def _training__(self):
        input = keras.Input((self.X_train.shape[1],))
        hidden = Dense(self.hidden_sizes[0], activation=self.activations[0])(input)
        output1 = Dense(1, activation=None)(hidden)
        output2 = Dense(1, activation=None)(input)
        output = Add()([output1, output2])
        self.model = keras.Model(input, output)

        # self.model = Sequential()
        # self.model.add(Dense(units=self.hidden_sizes[0], input_dim=self.X_train.shape[1], activation=self.activations[0]))
        # self.model.add(Dense(1))
        # self.model.add(Dense(units=1, input_dim=self.X_train.shape[1]))
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)
        sess = tf.Session(config=session_conf)
        K.set_session(session=sess)
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

        ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=self.print_train)
        self.loss_train = ml.history["loss"]


class FFNN1HL(RootMlnn):
    def __init__(self, root_base_paras=None, root_mlnn_paras=None):
        RootMlnn.__init__(self, root_base_paras, root_mlnn_paras)
        self.filename = "FFNN-1H-sliding_{}-net_para_{}".format(root_base_paras["sliding"], root_mlnn_paras)

    def _training__(self):
        self.model = Sequential()
        self.model.add(Dense(units=self.hidden_sizes[0], input_dim=self.X_train.shape[1], activation=self.activations[0]))
        session_conf = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)
        sess = tf.Session(config=session_conf)
        K.set_session(session=sess)
        self.model.add(Dense(1, activation=self.activations[1]))
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=self.print_train)
        self.loss_train = ml.history["loss"]

