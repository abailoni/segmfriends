try:
    import neuroglancer
except ImportError:
    neuroglancer = None

def visualize_zarr_in_neuroglancer(viewer, zarr_path, *layers_specs, local_fileserver_port=9000):
    """
    :param viewer: neuroglancer viewer
    :param zarr_path: relative path from home of the local web-file-server
    :param local_fileserver_port:
    :param layers_specs: Each arg should be a list (TODO: add support for dictionary?)
                with following items ("inner_path_zarr", layer_type, opt: "visualized_name")
                where layer_type can be either "image" or "segm"
    :return:
    """
    assert neuroglancer is not None, "Neuroglancer module is needed to use this function"
    assert isinstance(local_fileserver_port, int)
    # Remove initial slash, if present:
    zarr_path = zarr_path[1:] if zarr_path.startswith("/") else zarr_path

    # dimensions = neuroglancer.CoordinateSpace(
    #     scales=[1, 1, 1],
    #     units=['nm', 'nm', 'nm'],
    #     names=['x', 'y', 'z'])


    source_zarr = "zarr://http://127.0.0.1:{}/{}".format(local_fileserver_port, zarr_path)

    with viewer.txn() as s:
        for specs in layers_specs:
            assert isinstance(specs, (list, tuple))
            assert len(specs) == 2 or len(specs) == 3

            # Get layer specifications:
            zarr_inner_path = specs[0]
            layer_type = specs[1]
            if layer_type == "image":
                n_layer = neuroglancer.ImageLayer
            elif layer_type == "segm":
                n_layer = neuroglancer.SegmentationLayer
            else:
                raise ValueError("{} is not a recognised layer (only 'image' and 'segm' are accepted)")

            layer_name = specs[2] if len(specs) == 3 else zarr_inner_path
            s.layers[layer_name] = n_layer(source="{}/{}".format(source_zarr, zarr_inner_path))
