import sys
import re
import matplotlib.pyplot as plt

# Function to parse data and extract relevant information
def parse_data(description, data):
    # Extract base vector count and query vector count from description
    base_vector_count = int(re.search(r'(\d+) base', description).group(1))
    query_vector_count = int(re.search(r'(\d+) query', description).group(1))
    dimensions = int(re.search(r'dimensions (\d+)', description).group(1))
    dataset_name = re.search(r'/?(\S+): \d+ base and ', description).group(1).replace("/", "").replace(".", "")

    parsed_data = []
    line = " ".join(data)
    # Extract recall, query time, M and ef values
    recall = float(re.search(r'recall (\d+\.\d+) ', line).group(1))
    query_time = float(re.search(r'Query .* in (\d+\.\d+)s', line).group(1))
    M = int(re.search(r'M=(\d+)', line).group(1))
    ef = int(re.search(r'ef=(\d+)', line).group(1))
    disk_opt = re.search(r'PQ=(\w+)\s', line).group(1) == "true"
    pq = re.search(r'PQ@(\d+)\s', line).group(1) if disk_opt else "off"
    overquery = int(re.search(r'top \d+/(\d+) ', line).group(1))

    # Calculate throughput
    throughput = query_vector_count * 10 / query_time

    parsed_data.append((recall, throughput, M, ef, overquery, pq))
    
    return {
        'name': dataset_name, #+ "_" + str(M) + "_" + str(ef),
        'base_vector_count': base_vector_count,
        'dimensions': dimensions,
        'data': parsed_data
    }

def is_pareto_optimal(candidate, others):
    """Determine if a candidate point is Pareto-optimal."""
    for point in others:
        # Check if another point has higher or equal recall and throughput
        if point[0] >= candidate[0] and point[1] > candidate[1]:
            return False
        if point[0] > candidate[0] and point[1] >= candidate[1]:
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
    for recall, throughput, M, ef, overquery, pq in data:
        plt.scatter(recall, throughput, label=f'M={M}, ef={ef}, oq={overquery}, PQ={pq}')
        plt.annotate(f'M={M}, ef={ef}, oq={overquery}, PQ={pq}', (recall, throughput))
    
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
def group(parsed_datasets):
    grouped = dict()
    for val in parsed_datasets:
        key = val['name'] + "_" + str(val['base_vector_count']) + "_" + str(val['dimensions'])
        if key not in grouped:
            grouped[key] = {
                        'name': val['name'],
                        'base_vector_count': val['base_vector_count'],
                        'dimensions': val['dimensions'],
                        'data': []
                    }

        grouped[key]['data'].extend(val['data'])

    return list(grouped.values())


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

grouped_dataset = group(parsed_datasets)

# Filter and plot
for dataset in grouped_dataset:
    dataset['data'] = filter_pareto_optimal(dataset['data'])
for dataset in grouped_dataset:
    plot_dataset(dataset)

