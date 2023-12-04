# Copyright DataStax, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import sys
import re
import matplotlib.pyplot as plt

import re
from dataclasses import dataclass, astuple


@dataclass
class Point:
    pq: int
    recall: float
    throughput: float
    M: int
    ef: int
    overquery: int

def parse_data(description, data):
    """
    Parses a given set of data lines to extract relevant information.

    Parameters:
    - description (str): Metadata description of the dataset.
    - data (list of str): List of data lines to parse.

    Returns:
    - dict: A dictionary containing parsed information.
    """
    base_vector_count = int(re.search(r'(\d+) base', description).group(1))
    query_vector_count = int(re.search(r'(\d+) query', description).group(1))
    dimensions = int(re.search(r'dimensions (\d+)', description).group(1))
    dataset_name = re.search(r'(\S+):', description).group(1)

    parsed_data = []
    current_pq = None
    M = None
    for line in data:
        if "ProductQuantization" in line:
            current_pq = 'PQ@' + re.search(r'\((\d+)\)', line).group(1)
        elif "BinaryQuantization" in line:
            current_pq = 'BQ'
        elif "Uncompressed" in line:
            current_pq = 'UC'
        elif "Build M=" in line:
            M = int(re.search(r'M=(\d+)', line).group(1))
            ef = int(re.search(r'ef=(\d+)', line).group(1))
        elif "  Query " in line:
            if "(memory)" in line:
                # in-memory (on-heap) graph + vectors are benched as a sanity check;
                # we shouldn't include them in the plot of disk-based performance
                continue
            recall = float(re.search(r'recall (\d+\.\d+)', line).group(1))
            query_time = float(re.search(r'in (\d+\.\d+)s', line).group(1))
            overquery = int(re.search(r'top 100/(\d+) ', line).group(1))

            throughput = query_vector_count * 10 / query_time

            assert current_pq is not None
            assert M is not None
            parsed_data.append(Point(current_pq, recall, throughput, M, ef, overquery))

    return {
        'name': dataset_name,
        'base_vector_count': base_vector_count,
        'dimensions': dimensions,
        'data': parsed_data
    }


def is_pareto_optimal(candidate, others):
    """Determine if a candidate point is Pareto-optimal."""
    for point in others:
        # Check if another point has higher or equal recall and throughput
        if point.recall >= candidate.recall and point.throughput > candidate.throughput:
            return False
        if point.recall > candidate.recall and point.throughput >= candidate.throughput:
            return False
    return True

def filter_pareto_optimal(data):
    """Filter out only the Pareto-optimal points."""
    return [point for point in data if is_pareto_optimal(point, data)]

def plot_dataset(dataset, output_dir="."):
    # Extract dataset info
    name = dataset['name']
    base_vector_count = dataset['base_vector_count']
    dimensions = dataset['dimensions']
    data = dataset['data']
    
    # Create plot
    plt.figure(figsize=(15, 20))
    for pq, recall, throughput, M, ef, overquery in (astuple(p) for p in data):
        plt.scatter(recall, throughput, label=f'Q={pq}, M={M}, ef={ef}, oq={overquery}')
        plt.annotate(f'Q={pq}, M={M}, ef={ef}, oq={overquery}', (recall, throughput))
    
    # Set title and labels
    plt.title(f"Dataset: {name}\\nBase Vector Count: {base_vector_count}\\nDimensions: {dimensions}")
    plt.xlabel('Recall')
    plt.ylabel('Throughput')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Save the plot to a file
    filename = f"{output_dir}/{name}_plot.png"
    plt.savefig(filename)
    print("saved " + filename)
    
    # Clear the figure for the next plot
    plt.clf()

# Load and parse data
with open(sys.argv[1], 'r') as file:
    content = file.read().strip().split('\n\n')
datasets = []
for dataset in content:
    lines = dataset.split('\n')
    description = lines[0]
    data = lines[1:]
    datasets.append((description, data))
parsed_datasets = [parse_data(desc, data) for desc, data in datasets]

# Filter and plot
for dataset in parsed_datasets:
    dataset['data'] = filter_pareto_optimal(dataset['data'])
for dataset in parsed_datasets:
    plot_dataset(dataset)
