import json
import sys
import os

def combine_json_files(filenames):
    """
    Combine multiple JSON files into a single file.

    Args:
    filenames (list of str): List of filenames to read JSON data from.

    Returns:
    None: Writes the combined JSON data to 'results.json'.
    """
    combined_data = {}

    for filename in filenames:
        key = os.path.splitext(os.path.basename(filename))[0]
        try:
            with open(filename, 'r') as file:
                combined_data[key] = json.load(file)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    with open('results.json', 'w') as outfile:
        json.dump(combined_data, outfile, indent=4)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <file1.json> <file2.json> ...")
    else:
        combine_json_files(sys.argv[1:])
