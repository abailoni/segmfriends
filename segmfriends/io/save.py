import numpy as np
import vigra

from ..features import probs_to_costs, FeaturerLongRangeAffs
from ..utils import cantor_pairing_fct


def save_edge_indicators(affinities, segmentation, offsets, save_path,
                             n_threads=8, invert_affinities=False):
        featurer = FeaturerLongRangeAffs(offsets, n_threads=n_threads,
                                         invert_affinities=invert_affinities, return_dict=True)

        results = featurer(affinities, segmentation)
        rag = results['graph']
        # Edge indicators should be affinities (merge: 1.0; split: 0.0)
        edge_indicators = results['edge_indicators']
        edge_sizes = results['edge_sizes']

        print("Sorting edges...")
        uvIds = np.sort(rag.uvIds(), axis=1)
        cantor_coeff = cantor_pairing_fct(uvIds[:, 0], uvIds[:, 1])
        edge_data = np.stack((edge_indicators, edge_sizes), axis=-1)

        vigra.writeHDF5(edge_data, save_path, 'edge_data', compression='gzip')
        vigra.writeHDF5(cantor_coeff, save_path, 'cantor_ids', compression='gzip')


def save_edge_indicators_students(affinities, segmentation, offsets, save_path,
                         n_threads=8, invert_affinities=False):
    featurer = FeaturerLongRangeAffs(offsets, n_threads=n_threads,
                                     invert_affinities=invert_affinities, return_dict=True)

    results = featurer(affinities, segmentation)
    rag = results['graph']
    print("Number of nodes, edges:", rag.numberOfNodes, rag.numberOfEdges)
    # Edge indicators should be affinities (merge: 1.0; split: 0.0)
    edge_indicators = results['edge_indicators']
    edge_sizes = results['edge_sizes']

    costs = probs_to_costs(1-edge_indicators)


    print("Sorting edges...")
    uvIds = np.sort(rag.uvIds(), axis=1)

    vigra.writeHDF5(costs, save_path, 'edge_weights', compression='gzip')
    vigra.writeHDF5(uvIds, save_path, 'uv_IDs', compression='gzip')