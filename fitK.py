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
        if run['B'] == graph['N'] and run['K'] == 19:
            filtered_data.append((graph['N'], run['F']))

print(filtered_data)

# Extract N and F values
N_values, F_values = zip(*filtered_data)
N_values = np.array(N_values)
F_values = np.array(F_values)

# Define the function F = A + B * log(N)^X
def fit_function(N, A, B, X):
    return A + B * np.log(N)**X

# Fit the function to the data
bounds = (0, np.inf)
params, _ = curve_fit(fit_function, N_values, F_values, bounds=bounds)

# Extract the parameters A and X
A, B, X = params
print(A, B, X)

