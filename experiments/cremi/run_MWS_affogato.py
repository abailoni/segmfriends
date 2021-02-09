import numpy as np
from affogato.segmentation import compute_mws_segmentation



example_offsets = [
      [-1, 0, 0],
      [0, -1, 0],
      [0, 0, -1],
      [0, -4, 0],
      [0, 0, -4],
      [0, -12, 0],
      [0, 0, -12]
]

image_shape = (20,20,20)
affinities = np.random.uniform(size=(len(example_offsets),)+image_shape)

number_of_direct_neighbors_offsets = 3

MWS_segmentation = compute_mws_segmentation(affinities, example_offsets, number_of_direct_neighbors_offsets)


