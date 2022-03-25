#     crop_slice:
#       A:
#         - ":,:25"
#         - ":,25:50"
#         - ":,50:75"
#         - ":,75:100"
#         - ":,100:"
#       B:
#         - ":,:25, 90:, 580: 1900"
#         - ":,25:50, 90:, 580: 1900"
#         - ":,50:75, 90:, 580: 1900"
#         - ":,75:100, 90:, 580: 1900"
#         - ":,100:, 90:, 580: 1900"
#       C:
#         - ":,:25, 70:1450, 95:1425"
#         - ":,25:50, 70:1450, 95:1425"
#         - ":,50:75, 70:1450, 95:1425"
#         - ":,75:100, 70:1450, 95:1425"
#         - ":,100:, 70:1450, 95:1425"
#
# #    sub_crop_slice: ":,:, 200:250, 200:250"
#     sub_crop_slice: ":,:,:,:"
from elf.evaluation import cremi_score
from elf.segmentation.gasp_utils import find_indices_direct_neighbors_in_offsets
from segmfriends.utils import readHDF5

long_range_prob = 0.1
crop_slice = "50:60, 90:, 580: 1900"
affs = readHDF5("/scratch/bailoni/datasets/GASP/SOA_affs/sampleB_train.h5", 'predictions/full_affs', crop_slice=":," + crop_slice)
GT = readHDF5("/scratch/bailoni/datasets/GASP/SOA_affs/sampleB_train.h5", 'segmentations/groundtruth_fixed_OLD', crop_slice=crop_slice)

offsets = [
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

import numpy as np

# import napari for data visualisation
import napari

# Import function to download the example data
from elf.segmentation.utils import load_mutex_watershed_problem

# import the segmentation functionality from elf
from elf.segmentation import GaspFromAffinities
from elf.segmentation import run_GASP

# Import an utility function from nifty that we will need to generate a toy graph:
from nifty.graph import UndirectedGraph

# import the open_file function from elf, which supports opening files
# in hdf5, zarr, n5 or knossos file format
from elf.io import open_file


affs += np.random.normal(scale=1e-4, size=affs.shape)
# affs = np.clip(affs, 0., 1.)

# Sample some long-range edges:
edge_mask = np.random.random(
    affs.shape) < long_range_prob
# Direct neighbors should be always added:
is_offset_direct_neigh, _ = find_indices_direct_neighbors_in_offsets(offsets)
edge_mask[is_offset_direct_neigh] = True

kwargs = {
    "Mean": {'linkage_criteria': 'average'},
    "Median": {'linkage_criteria': 'quantile',
               'linkage_criteria_kwargs': {'q': 0.5, 'numberOfBins': 40}},
    "MWS_Grid": {'linkage_criteria': 'mutex_watershed',
                 'use_efficient_implementations': True},
    "MWS_Eff_graph": {'linkage_criteria': 'mutex_watershed',
                      'use_efficient_implementations': False,
                      'force_efficient_graph_implementation': True},
    "MWS_GASP": {'linkage_criteria': 'mutex_watershed',
                 'use_efficient_implementations': False},
}

results = []

for name in kwargs:
    run_kwargs = kwargs[name]
    # Run the algorithm:
    gasp_instance = GaspFromAffinities(offsets,
                                       set_only_direct_neigh_as_mergeable=False,
                                       run_GASP_kwargs=run_kwargs)
    # To speed-up computations, here we use only part of the example data:
    segmentation, runtime = gasp_instance(affs, mask_used_edges=edge_mask)
    scores = cremi_score(segmentation, GT, ignore_gt=[0])
    results.append([name, runtime] + list(scores))
    print(name, runtime, scores)


import pandas
df = pandas.DataFrame(data=results, columns=["Name", "Runtime", "VI-split", "VI-merge", "ARAND", "Cremi-Score"])
df.to_csv("/scratch/bailoni/projects/gasp/runtimes.csv", index=False)
