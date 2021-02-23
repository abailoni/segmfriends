from scipy.ndimage import zoom
import numpy as np
import os

from . import various as segm_utils


try:
    import cremi
    from cremi.evaluation import NeuronIds
    from cremi import Volume
    from cremi.io import CremiFile
except ImportError:
    cremi = None

try:
    import cremi_tools
    from cremi_tools.alignment import backalign_segmentation
    from cremi_tools.alignment.backalign import bounding_boxes as magic_bboxes
except ImportError:
    cremi_tools = None

# TODO: pass as argument of the function?
shape_padded_aligned_datasets = {
    "A+": (200, 3727, 3505),
    "B+": (200, 3832, 5455),
    "C+": (200, 3465, 3668)
}

def prepare_submission(sample, path_segm, inner_path_segm,
                       path_bbox_slice, ds_factor=None):
    """
    :param path_bbox_slice: path to the csv file
    :param ds_factor: for example (1, 2, 2)
    """
    assert cremi is not None, "cremi package is needed to prepare the submission " \
                              "(https://github.com/constantinpape/cremi_python.git, branch `py3`)"
    assert cremi_tools is not None, "cremi_tools package is needed to prepare the submission" \
                                    "(https://github.com/constantinpape/cremi_tools)"

    segm = segm_utils.readHDF5(path_segm, inner_path_segm)

    bbox_data = np.genfromtxt(path_bbox_slice, delimiter=';', dtype='int')
    assert bbox_data.shape[0] == segm.ndim and bbox_data.shape[1] == 2
    # bbox_slice = tuple(slice(b_data[0], b_data[1]) for b_data in bbox_data)

    if ds_factor is not None:
        assert len(ds_factor) == segm.ndim
        assert all(fct >= 1. for fct in ds_factor), "Zoom function only works properly for upscaling, not downloscaling"
        segm = zoom(segm, ds_factor, order=0)

    padding = tuple((slc[0], shp - slc[1]) for slc, shp in zip(bbox_data, shape_padded_aligned_datasets[sample]))
    padded_segm = np.pad(segm, pad_width=padding, mode="constant")

    # Apply Constantin crop and then backalign:
    cropped_segm = padded_segm[magic_bboxes[sample]]
    tmp_file = path_segm.replace(".h5", "_submission_temp.hdf")
    backalign_segmentation(sample, cropped_segm, tmp_file,
                           key="temp_data",
                           postprocess=False)

    # Create a CREMI-style file ready to submit:
    final_submission_path = path_segm.replace(".h5", "_submission.hdf")
    file = CremiFile(final_submission_path, "w")

    # Write volumes representing the neuron and synaptic cleft segmentation.
    backaligned_segm = segm_utils.readHDF5(tmp_file, "temp_data")
    neuron_ids = Volume(backaligned_segm.astype('uint64'), resolution=(40.0, 4.0, 4.0),
                        comment="")

    file.write_neuron_ids(neuron_ids)
    file.close()

    os.remove(tmp_file)
