import json
import sys
import numpy as np
from scipy.optimize import curve_fit

data = json.load(sys.stdin)

# Extract the relevant dataset
dataset = data['intfloat_e5-small-v2_100000']

# dataset looks like this
# [{
#     "N": 2048,
#     "data": [
#         {
#             "K": 1,
#             "B": 20,
#             "F": 807
#         },
#         ...
#    ]
# },
# {
#     "N": 4096,
#     ...

# extract values where K=1 and B=N
filtered_data = []
for graph in dataset:
    for run in graph['data']:
        if run['B'] == graph['N']:
            filtered_data.append((graph['N'], run['K'], run['F']))

print(filtered_data[:10])

# Extract N and F values
N_values, K_values, F_values = zip(*filtered_data)
N_values = np.array(N_values)
K_values = np.array(K_values)
F_values = np.array(F_values)
combined_variables = np.vstack((N_values, K_values)).T

# Define the function F = ...
def fit_function(variables, A, B, X, Y):
    N, K = variables.T
    return A + B * np.log(N)**X * K**Y

# Fit the function to the data
bounds = (0, np.inf)
params, _ = curve_fit(fit_function, combined_variables, F_values, bounds=bounds)

# Extract the parameters
A, B, X, Y = params
print(A, B, X, Y)
