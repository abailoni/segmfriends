"""
##########
# Arguments of function compute_affinities()
##########

labels: 2D or 3D numpy array of type int64, uint64 or bool

offset: List of 2D or 3D offsets indicating the neighborhood pattern. For example (in the 3D case):
            [
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
                [0, -4, 0],
                [0, 0, -4],
                [0, -10, -10],
            ]

have_ignore_label: (boolean) indicating whether there is a label that should be ignored while computing the affinities.
                    By default is False.

ignore_label: (int) value of the ignore label. By default is 0. If `have_ignore_label` is False, is ignored.


##########
# Outputs:
##########

affinities: boolean numpy array. It will have shape ( len(offset), ) + labels.shape

valid_mask: boolean numpy array with the same shape of `affinities`, indicating which computed affinities
            are valid and which are not (for example because they go out of the segmentation boundaries or they
            involve the ignore_label).
"""

import numpy as np
from affogato.affinities import compute_affinities

example_offsets = [
      [-1, 0, 0],
      [0, -1, 0],
      [0, 0, -1],
      [0, -4, 0],
      [0, 0, -4],
      [0, -12, 0],
      [0, 0, -12]
]

test_shape = (20,20,20)
example_segmentation = np.random.randint(0,1000, size=test_shape)

affinities, valid_affinities = compute_affinities(example_segmentation.astype('uint64'), example_offsets, False, 0)
