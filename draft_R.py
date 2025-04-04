import os
import pandas as pd

import numpy as np
#from pandas import DataFrame
#from torch.nn import Bilinear
#from torch.nn.functional import upsample

from sklearn.datasets import make_regression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras import Model
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
import keras
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
data_train = pd.read_excel('state2.xlsx', header=None)
correct_train = pd.read_excel('obser1.xlsx', header=None)
data_train = np.array(data_train)
correct_train = np.array(correct_train)
data_train, data_test, correct_train, correct_test = train_test_split(data_train, correct_train, test_size=0.2)
#print("After reshape test data set shape:\n", test_data_reshaped.shape)
from keras import Sequential
from keras.layers import Lambda
import tensorflow as tf
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
# Scale the input data
scaler = StandardScaler()
X_train = scaler.fit_transform(data_train)
X_test = scaler.transform(data_test)
# Define the grid search parameters
param_grid = {'hidden_layers': [1,2, 3, 4],
              'nodes': [5, 10, 15,20],
              'batch_size': [8, 16, 32],

              'epochs': [700,800,900,1000],
              'optimizer': ['adam', 'sgd'],
              'init_mode': ['uniform', 'normal', 'random_normal'],
              'activation': ['Sigmoid', 'ReLU', 'Softmax']
              }
# Add the first layer of nodes
param_grid['nodes'][0] = [5, 10, 15]
# Define the model

def create_model(hidden_layers=1,nodes=10, optimizer='adam',init_mode='uniform',activation='relu'):
  nodes_in_first_layer = [5, 10, 15]
  model = Sequential()
  model.add(Dense(nodes_in_first_layer[0], input_dim=X_train.shape[1], activation=activation,kernel_initializer=init_mode))
  for i in range(hidden_layers - 1):
    model.add(Dense(nodes, activation='relu'))
    model.add(Dense(4))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model
# Create the model
from keras.wrappers.scikit_learn import KerasRegressor
model = KerasRegressor(build_fn=create_model)
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
  # Perform the grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', verbose=2)
grid_result = grid.fit(X_train, correct_train)
# Print the best parameters and best score
from sklearn.metrics import mean_squared_error
print("Best parameters: ", grid_result.best_params_)
print("Best score: ", grid_result.best_score_)

# Evaluate the final model on the test set
best_model = grid_result.best_estimator_
y_pred = best_model.predict(X_test)
test_score = np.sqrt(mean_squared_error(correct_test, y_pred))
print("Test score: ", test_score)
with open("output3.txt", "w") as file:
    file.write(str(grid_result.best_params_))
with open("output4.txt", "w") as file:
    file.write(str(grid_result.best_score_))
with open("output5.txt", "w") as file:
    file.write(str(test_score))