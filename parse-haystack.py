import json
import re
import sys

# Usage:
#   python parse-haystack.py < haystack.txt > haystack.json

def parse_file(file_lines):
    # Regular expressions for matching dataset introduction and run lines
    dataset_regex = r'^(.*?):\s*\d+\s*base and \d+\s*query vectors loaded, dimensions \d+'
    run_regex = r'^Build N=(\d+) M=\d+ ef=\d+ in \S+s with \S+ short edges'
    data_line_regex = r'^Looking for top (\d+) of (\d+) ordinals required visiting (\d+) nodes'

    parsed_data = {}
    current_dataset = None
    current_run = None

    for line in file_lines:
        # Check for dataset introduction
        dataset_match = re.match(dataset_regex, line)
        if dataset_match:
            current_dataset = dataset_match.group(1)
            parsed_data[current_dataset] = []
            continue

        # Check for run start
        run_match = re.match(run_regex, line)
        if run_match and current_dataset is not None:
            current_run = {'N': int(run_match.group(1)), 'data': []}
            parsed_data[current_dataset].append(current_run)
            continue

        # Check for data line
        data_line_match = re.match(data_line_regex, line)
        if data_line_match and current_run is not None:
            K, B, F = map(int, data_line_match.groups())
            current_run['data'].append({'K': K, 'B': B, 'F': F})

    return parsed_data

# Read the file contents from stdin
file_contents = sys.stdin.readlines()
json_data = parse_file(file_contents)
print(json.dumps(json_data, indent=2))
