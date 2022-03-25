import argparse
import time
import os

import h5py
import pandas as pd
import numpy as np

from elf.evaluation import cremi_score
from elf.segmentation.gasp_utils import find_indices_direct_neighbors_in_offsets
from segmfriends.utils import readHDF5
from elf.segmentation import GaspFromAffinities

# The example data from https://oc.embl.de/index.php/s/sXJzYVK0xEgowOz
PATH = "/scratch/bailoni/datasets/GASP/SOA_affs/sampleB_train.h5"

long_range_prob = 0.1


OFFSETS = [
  [-1, 0, 0],
  [0, -1, 0],
  [0, 0, -1],
  [-2, 0, 0],
  [0, -3, 0],
  [0, 0, -3],
  [-3, 0, 0],
  [0, -9, 0],
  [0, 0, -9],
  [-4, 0, 0],
  [0, -27, 0],
  [0, 0, -27]
]


kwargs = {
    # "Mean": {'linkage_criteria': 'average'},
    # "Median": {'linkage_criteria': 'quantile',
    #            'linkage_criteria_kwargs': {'q': 0.5, 'numberOfBins': 40}},
    "MWS_Grid": {'linkage_criteria': 'mutex_watershed',
                 'use_efficient_implementations': True},
    "MWS_Eff_graph": {'linkage_criteria': 'mutex_watershed',
                      'use_efficient_implementations': False,
                      'force_efficient_graph_implementation': True},
    "MWS_GASP": {'linkage_criteria': 'mutex_watershed',
                 'use_efficient_implementations': False},
}

METHOD = None



def measure_runtime(affs, n):
    times = []
    for _ in range(n):

        affs += np.random.normal(scale=1e-4, size=affs.shape)

        # Sample some long-range edges:
        edge_mask = np.random.random(
            affs.shape) < long_range_prob
        # Direct neighbors should be always added:
        is_offset_direct_neigh, _ = find_indices_direct_neighbors_in_offsets(OFFSETS)
        edge_mask[is_offset_direct_neigh] = True


        t = time.time()
        run_kwargs = kwargs[METHOD]
        # Run the algorithm:
        gasp_instance = GaspFromAffinities(OFFSETS,
                                           set_only_direct_neigh_as_mergeable=False,
                                           run_GASP_kwargs=run_kwargs)
        gasp_instance(affs, mask_used_edges=edge_mask)
        times.append(time.time() - t)
        # scores = cremi_score(segmentation, GT, ignore_gt=[0])
        # results.append([name, runtime] + list(scores))
        # print(name, runtime, scores)

    return np.min(times)


def increase_shape(shape, full_shape, axis, is_full):
    if all(is_full):
        return shape, axis, is_full
    if is_full[axis]:
        axis = (axis + 1) % 3
        return increase_shape(shape, full_shape, axis, is_full)

    shape[axis] *= 2
    if shape[axis] >= full_shape[axis]:
        is_full[axis] = True
        shape[axis] = full_shape[axis]
    axis = (axis + 1) % 3
    return shape, axis, is_full


def benchmark(path, out_path, n):
    with h5py.File(path, "r") as f:
        affs = f["predictions/full_affs"][:,50:60, 200:500, 750:1050]
    assert affs.shape[0] == len(OFFSETS)

    full_shape = affs.shape[1:]
    shape = [4, 64, 64]

    results = []
    axis = 0
    is_full = [False] * 3

    while True:
        bb = (slice(None),) + tuple(slice(0, sh) for sh in shape)
        affs_ = affs[bb]
        t = measure_runtime(affs_, n)
        print("Run benchmark for", shape, "in t", t, "[s]")
        str_shape = "-".join(map(str, shape))
        size = np.prod(list(shape))
        results.append([str_shape, size, t])

        shape, axis, is_full = increase_shape(shape, full_shape, axis, is_full)
        if all(is_full):
            break

    t = measure_runtime(affs, n)
    print("Run benchmark for", full_shape, "in t", t, "[s]")
    str_shape = "-".join(map(str, full_shape))
    size = np.prod(list(full_shape))
    results.append([str_shape, size, t])

    results = pd.DataFrame(results, columns=["shape", "size", "time [s]"])
    results.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument("-o", "--output_path", required=True)  # where to save the csv with results
    parser.add_argument("-p", "--path", default=PATH)
    parser.add_argument("-n", default=5, type=int)
    args = parser.parse_args()

    for name in kwargs:
        global METHOD
        METHOD = name
        out_path = os.path.join("/scratch/bailoni/projects/gasp/MWS_benchmark", "benchmark_{}.csv".format(name))
        benchmark(args.path, out_path, args.n)


if __name__ == "__main__":
    main()
