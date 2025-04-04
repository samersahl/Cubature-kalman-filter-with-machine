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
data_train = pd.read_excel('state1.xlsx', header=None)
correct_train = pd.read_excel('obser1.xlsx', header=None)
P= pd.read_excel('PP2.xlsx', header = None)
print(data_train.shape)








