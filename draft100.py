from itertools import product

# Define the grid search parameters
param_grid = {'hidden_layers': [2, 3, 4],
              'nodes': [20,30,40],
              'batch_size': [8, 16, 32],
              'epochs': [500, 600, 700],
              'optimizer': ['adam', 'sgd'],
              'init_mode': ['uniform', 'normal', 'random_normal'],
              'activation':['Sigmoid','ReLU','Softmax']
              }

# Add the first layer of nodes

# Calculate the total number of parameter combinations
num_combinations = len(list(product(*param_grid.values())))
print("Total number of parameter combinations: ", num_combinations)

