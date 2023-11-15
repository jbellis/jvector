import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys


# Parse a single line and extract K, B, and N
def parse_line(line):
    match = re.search(r'Looking for top (\d+) of (\d+) ordinals required visiting (\d+) nodes', line)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None


def parse_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parsed = parse_line(line)
            if parsed:
                data.append(parsed)
    return pd.DataFrame(data, columns=['K', 'B', 'N'])


def create_3d_bar_chart(df):
    # Sort the DataFrame by B in ascending order
    df_sorted = df.sort_values(by='B')

    # Extract sorted values for plotting
    x = df_sorted['K']
    y = df_sorted['B']
    z = np.zeros_like(x)
    dy = np.ones_like(x) * 0.1  # Width of the bars
    dx = dy * 50
    dz = df_sorted['N']

    # Calculate the logarithm of B for bar heights
    log_b = np.log(y)
    norm = plt.Normalize(log_b.min(), log_b.max())
    colors = plt.cm.twilight(norm(log_b))

    # Create and display the plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the bars
    for xi, yi, zi, dzi, color in zip(x, log_b, z, dz, colors):
        ax.bar3d(xi, yi, zi, dx, dy, dzi, color=color)

    # Invert the B axis here
    ax.set_ylim(ax.get_ylim()[::-1])

    # Set custom y-axis ticks to match original B values with 'k' abbreviation
    y_tick_values = [1000, 10000, 100000]  # B values to display
    y_tick_positions = np.log(y_tick_values)  # Compute their logarithms
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels([f'{int(value/1000)}k' for value in y_tick_values])

    # Set custom z-axis ticks with 'k' abbreviation
    z_tick_values = [50000, 100000, 150000, 200000, 250000]  # N values to display
    ax.set_zticks(z_tick_values)
    ax.set_zticklabels([f'{int(value/1000)}k' for value in z_tick_values])

    ax.set_xlabel('Top K')
    ax.set_ylabel('B candidates')
    ax.set_zlabel('N nodes visited')
    ax.set_title('Nodes visited while searching for top K out of B candidate values\n(NYTimes dataset of 290k vectors)')
    plt.show()


df = parse_file(sys.argv[1])
create_3d_bar_chart(df)
