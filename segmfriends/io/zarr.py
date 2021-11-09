import zarr
import os
import numpy as np
import shutil

def load_array_from_zarr_group(z_path,
                               inner_path,
                               z_slice=None,
                               apply_valid_mask=False,
                               valid_mask_name="valid_mask",
                               crop_slice=None):
    # TODO: add support for crop slice; rename mask
    assert crop_slice is None, "Not implemeted yet"

    assert os.path.exists(z_path), "Zarr file does not exist: {}".format(z_path)
    z_group = zarr.open(z_path, mode="w+")
    assert inner_path in z_group, "Inner dataset {} not found in zarr group {}".format(inner_path, z_path)

    # Load dataset in memory:
    dataset = z_group[inner_path][:]
    if z_slice is not None:
        # TODO: generalize
        assert isinstance(z_slice, int)
        dataset = dataset[z_slice]

    if apply_valid_mask:
        assert valid_mask_name in z_group
        # TODO: generalize
        assert z_slice is not None
        original_image_shape = tuple(z_group[valid_mask_name][:][z_slice])
        original_crop = tuple(slice(shp) for shp in original_image_shape)
        dataset = dataset[original_crop]

    return dataset

def delete_datasets_in_zarr_group(z_path, *datasets_to_delete):
    assert os.path.exists(z_path), "Zarr file does not exist: {}".format(z_path)

    # Delete datasets from zarr hierarchy:
    for data_name in datasets_to_delete:
        data_path = os.path.join(z_path, data_name)
        if os.path.exists(data_path):
            shutil.rmtree(data_path)


def append_arrays_to_zarr(z_path,
                          add_array_dimensions=False,
                          overwrite_z_group=False,
                          delete_previous_inner_datasets=False,
                          keep_valid_mask=False,
                          valid_mask_name="valid_mask",
                          **datasets):
    # TODO: update mask name...

    assert isinstance(z_path, str)
    assert len(datasets)
    if overwrite_z_group:
        shutil.rmtree(z_path)

    create_new_datasets = not os.path.exists(z_path)
    if delete_previous_inner_datasets:
        create_new_datasets = True
        assert not keep_valid_mask, "It is not possible to save a valid mask when only part of the datasets are rewritten, " \
                                    "because the mask would not be consistent across the zarr group"
        for key in datasets:
            shutil.rmtree(os.path.join(z_path, key))

    if create_new_datasets:
        expanded_datasets = {key: datasets[key][None] for key in datasets}

        if keep_valid_mask:
            sample_key = [key for key in datasets][0]
            expanded_datasets[valid_mask_name] = np.array([list(datasets[sample_key].shape)])

        zarr.save_group(z_path, **expanded_datasets)

        if add_array_dimensions:
            z_group = zarr.open(z_path, mode='w+')
            for key in z_group:
                z_group[key].attrs['_ARRAY_DIMENSIONS'] = ["z", "y", "x"]
    else:
        z_group = zarr.open(z_path, mode="w+")

        old_valid_mask_shape = None

        # Expand zarr arrays:
        for key, array in datasets.items():
            if key in z_group:
                z_array = z_group[key]
                new_shape = list(z_array.shape[1:])
                arr_shape = array.shape
                for i, img_shp, z_shp in zip(range(len(arr_shape)),
                                             arr_shape,
                                             z_array.shape[1:]):
                    if img_shp > z_shp:
                        new_shape[i] = img_shp

                zarr_slice = (-1,) + tuple(slice(0, shp) for shp in arr_shape)

                if keep_valid_mask:
                    assert valid_mask_name in z_group

                    if old_valid_mask_shape is None:
                        image_shapes = z_group[valid_mask_name]
                        old_valid_mask_shape = z_array.shape[1:]
                        image_shapes.resize((image_shapes.shape[0] + 1, image_shapes.shape[1]))
                        # Append new valid mask:
                        image_shapes[-1] = np.array(z_array.shape[1:])

                    assert old_valid_mask_shape == z_array.shape[1:], "Not all datasets have the same shape. " \
                                                                  "Valid mask cannot be used in this case"

                z_array.resize((z_array.shape[0] + 1,) + tuple(new_shape))

                # Append new data:
                z_array[zarr_slice] = array
            else:
                assert not keep_valid_mask, "A new dataset was created from scratch in the zarr hierarchy." \
                                            "To use a valid mask and avoid inconsistencies, set overwrite_z_group to" \
                                            "True and delete any previous content in the zarr group"

                # Add new dataset to the zarr hierarchy:
                z_group.create_dataset(key, data=array[None], shape=(1,)+array.shape)

                if add_array_dimensions:
                    z_group[key].attrs['_ARRAY_DIMENSIONS'] = ["z", "y", "x"]
