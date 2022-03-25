import argparse
import os

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib import rc


def parse_result(path):
    tab = pd.read_csv(path)[["size", "time [s]"]]
    return tab


def plot_benchmark(inputs):
    # rc('text', usetex=True)
    matplotlib.rcParams.update({'font.size': 17})
    plt.figure(figsize=(10, 5))

    results = []
    names = []
    for inp in inputs:
        names.append(os.path.splitext(os.path.split(inp)[1])[0])
        results.append(parse_result(inp))

    for name, res in zip(names, results):
        labels = {
            "benchmark_MWS_Eff_graph": "Efficient graph implementation",
            "benchmark_MWS_GASP": "Naive graph implementation",
            "benchmark_MWS_Grid": "Pixel-grid implementation",
        }
        test = sns.lineplot(data=res, x="size", y="time [s]", label=labels[name])
    plt.xlabel('Number of Nodes')
    plt.ylabel('Runtime [s]')
    plt.tight_layout()
    plt.savefig("/scratch/bailoni/projects/gasp/MWS_benchmark/runtime-MWS.pdf")


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--inputs", "-i", required=True, nargs="+")
    # args = parser.parse_args()
    csv_dir = "/scratch/bailoni/projects/gasp/MWS_benchmark"
    files = [
        "benchmark_MWS_GASP.csv",
        "benchmark_MWS_Eff_graph.csv",
        "benchmark_MWS_Grid.csv"
    ]
    plot_benchmark([os.path.join(csv_dir, f) for f in files])\



if __name__ == "__main__":
    main()
