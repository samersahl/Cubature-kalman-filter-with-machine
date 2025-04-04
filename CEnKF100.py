import time
import os
import pandas as pd
from keras import Model
import numpy as np
from keras.callbacks import Callback, ModelCheckpoint
#from pandas import DataFrame
#from torch.nn import Bilinear
#from torch.nn.functional import upsample
from keras.models import load_model
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
import keras
#import torch
#import torch.nn as nn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from keras.layers.convolutional import UpSampling1D
import numpy as np
#import scipy.io as spio
from sklearn.model_selection import train_test_split
import pandas as pd
#from sklearn.metrics import mean_squared_log_error
#import plot_keras_history as plt
#import matplotlib.pyplot as p
#import torch.optim as optim
from scipy import io
data_train = pd.read_excel('state.xlsx', header=None)
correct_train = pd.read_excel('obser.xlsx', header=None)
P= pd.read_excel('PP.xlsx', header = None)
data_train = np.array(data_train)
data_train=data_train.transpose()
correct_train = np.array(correct_train)
correct_train =correct_train.transpose()
P_k = np.array(P)
P_k=np.linalg.inv(P_k)
sample_size = data_train.shape[0] # 40
time_steps = data_train.shape[1] #100
input_dimension = 1
train_data_reshaped = data_train.reshape(sample_size, time_steps, input_dimension)
correct_data_reshaped = correct_train.reshape(sample_size, time_steps, input_dimension)
P_k=P_k.reshape(time_steps, time_steps, input_dimension)
data_train, data_test, correct_train, correct_test = train_test_split(data_train, correct_train, test_size=0.2)
test_data_reshaped = data_test.reshape(data_test.shape[0], data_test.shape[1], input_dimension)
test_correct_reshaped = correct_test.reshape(correct_test.shape[0], correct_test.shape[1], input_dimension)
#print("After reshape test data set shape:\n", test_data_reshaped.shape)
from keras import Sequential
from keras.layers import Lambda
import tensorflow as tf
import torch
rmse = tf.keras.metrics.RootMeanSquaredError()

import torch

import torch
def custom_loss(correct_data_reshaped, y_pred):
    P = pd.read_excel('PP.xlsx', header=None)
    P_k = np.array(P)
    A = np.linalg.inv(P_k)
    l1 = ( y_pred - correct_data_reshaped)
    l1t = tf.transpose(l1, perm=[0, 2, 1])
    l = tf.matmul(tf.matmul(l1t, A), l1)
    m_l = tf.reduce_mean(l)
    return m_l


def build_multi_conv1D_model():
    n_timesteps = train_data_reshaped.shape[1]  # 40
    n_features = train_data_reshaped.shape[2]  # 1
    input = keras.layers.Input(shape=(n_timesteps, n_features))
    batchnorm = keras.layers.BatchNormalization()(input)
    cnn1 = keras.layers.Conv1D(25, 5, kernel_initializer='uniform',
                               bias_initializer='zeros', padding="same",
                               activation="relu", strides=1)(batchnorm)
    multi = keras.layers.Multiply()([cnn1, cnn1])
    conc = keras.layers.concatenate([cnn1, multi], axis=-1)
    cnn2 = keras.layers.Conv1D(40, 5, padding="same", activation="relu", strides=1)(conc)
    cnn3 = keras.layers.Conv1D(1, 1, padding="same", activation="linear", strides=1, kernel_regularizer=keras.regularizers.L2(.0001))(cnn2)
    add = keras.layers.Add()([cnn3, input])

    model = Model(input, add)
    loss_fn = custom_loss

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=[rmse, 'mae'])

    return model





model_conv1D_2 = build_multi_conv1D_model()
model_conv1D_2.summary()
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

from keras.utils.vis_utils import plot_model

print(plot_model(model_conv1D_2, to_file='model_plot.png', show_shapes=True, show_layer_names=True))
model_fit = model_conv1D_2.fit(train_data_reshaped, correct_data_reshaped,
                               validation_data=(test_data_reshaped, test_correct_reshaped),
                               epochs=700, verbose=2, batch_size=8, validation_batch_size=2)

Model_Outputs = {f'layer{idx}':layer.output for idx,layer in enumerate(model_conv1D_2.layers)}
Model_Outputs
prediction_whole_model = model_conv1D_2.predict(train_data_reshaped)
prediction_whole_model[0].flatten()


last_cnn_model = Model(model_conv1D_2.input, Model_Outputs['layer6'])
predictions_last_cnn = last_cnn_model.predict(train_data_reshaped)
predictions_last_cnn[0].flatten()
last_cnn_model = Model(model_conv1D_2.input, Model_Outputs['layer6'])
predictions_last_cnn = last_cnn_model.predict(train_data_reshaped)
predictions_last_cnn[0].flatten()
print(model_fit.history.keys())
length = len(model_fit.history['loss'])

import plotly.graph_objects as go
# Create traces
fig = go.Figure()
for tr in model_fit.history.keys():
  fig.add_trace(go.Scatter(y=model_fit.history[tr], x=list(range(length)),
                    mode='lines+markers',
                    name=tr))
# Calculate the loss

io.savemat('x1.mat', {"data1":  prediction_whole_model})
fig.show()







