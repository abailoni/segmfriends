import numpy as np
import vigra
import getpass
import socket

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

def get_hci_home_path():
    username = getpass.getuser()
    hostname = socket.gethostname()
    if hostname == 'trendytukan' and username == 'abailoni':
        return '/net/hciserver03/storage/abailoni/'
    elif hostname == 'trendytukan' and username == 'abailoni_local':
        return '/home/abailoni_local/hci_home/'
    elif hostname == 'ialgpu01':
        return '/home/abailoni/hci_home/'
    else:
        return '/net/hciserver03/storage/abailoni/'

def get_trendytukan_drive_path():
    username = getpass.getuser()
    hostname = socket.gethostname()
    if hostname == 'trendytukan' and username == 'abailoni':
        return '/mnt/localdata0/abailoni/'
    elif hostname == 'trendytukan' and username == 'abailoni_local':
        return '/home/abailoni_local/trendyTukan_localdata0/'
    else:
        raise ValueError("Local drive not accessible from centrally administred machines!")
