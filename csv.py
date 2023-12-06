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
        if run['K'] == 1:
            filtered_data.append((graph['N'], run['B'], run['F']))

for t in filtered_data:
    print('%s,%s,%s' % t)

