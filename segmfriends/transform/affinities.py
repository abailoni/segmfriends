"""
Custom version of the methods in neurofire
(can handle boundary pixels and glia_masks)
"""
import numpy as np

try:
    # from inferno.io.core import ZipReject, Concatenate
    from inferno.io.transform import Compose, Transform
    from inferno.io.transform.generic import AsTorchBatch
    from inferno.io.transform.volume import RandomFlip3D, VolumeAsymmetricCrop
    from inferno.io.transform.image import RandomRotate, ElasticTransform
except ImportError:
    raise ImportError("Couldn't find 'inferno' module, affinity calculation is not available")

try:
    from neurofire.datasets.loader import RawVolume, SegmentationVolume, RawVolumeWithDefectAugmentation
    from neurofire.transform.artifact_source import RejectNonZeroThreshold
    from neurofire.transform.affinities import Segmentation2Affinities2or3D, Segmentation2Affinities
    from neurofire.transform.volume import RandomSlide
except ImportError:
    raise ImportError("Couldn't find 'neurofire' module, affinity calculation is not available")

try:
    from affogato.affinities import compute_affinities
except ImportError:
    compute_affinities = None

def affinity_config_to_transform(**affinity_config):
    assert 'offsets' in affinity_config, "Need 'offsets' parameter in affinity config"
    return Segmentation2AffinitiesPluGliaAndBoundary(**affinity_config)


class Segmentation2AffinitiesPluGliaAndBoundary(Segmentation2Affinities2or3D):
    def __init__(self, offsets, dtype='float32',
                 retain_mask=False, ignore_label=None,
                 boundary_label=None,
                 glia_label=None,
                 retain_glia_mask=False,
                 train_affs_on_glia=False,
                 retain_extra_masks=False,
                 retain_segmentation=False, segmentation_to_binary=False,
                 map_to_foreground=True, learn_ignore_transitions=False,
                 **super_kwargs):
        assert compute_affinities is not None,\
            "Couldn't find 'affogato' module, affinity calculation is not available"
        assert isinstance(offsets, (list, tuple)), "`offsets` must be a list or a tuple."
        super(Segmentation2Affinities2or3D, self).__init__(**super_kwargs)
        self.dim = len(offsets[0])
        assert self.dim in (2, 3), str(self.dim)
        assert all(len(off) == self.dim for off in offsets[1:])
        self.offsets = offsets
        self.dtype = dtype
        self.retain_mask = retain_mask
        self.ignore_label = ignore_label
        self.boundary_label = boundary_label
        self.glia_label = glia_label
        self.train_affs_on_glia = train_affs_on_glia
        self.retain_glia_mask = retain_glia_mask
        self.retain_extra_masks = retain_extra_masks
        self.retain_segmentation = retain_segmentation
        self.segmentation_to_binary = segmentation_to_binary
        assert not (self.retain_segmentation and self.segmentation_to_binary),\
            "Currently not supported"
        self.map_to_foreground = map_to_foreground
        self.learn_ignore_transitions = learn_ignore_transitions


    def input_function(self, tensor):
        labels = tensor[0]
        boundary_mask = None
        glia_mask = None
        extra_masks = None

        if tensor.shape[0] > 1:
            # Here we get both the segmentation and an additional mask:
            assert tensor.shape[0] == 2, "Only one additional mask is supported at the moment"
            extra_masks = tensor[1]

            if self.boundary_label is not None:
                boundary_mask = (extra_masks == self.boundary_label)
            if not self.train_affs_on_glia and self.glia_label is not None:
                glia_mask = (extra_masks == self.glia_label)

        output, mask = compute_affinities(labels.astype('int64'), self.offsets,
                             ignore_label=self.ignore_label,
                             boundary_mask=boundary_mask,
                             glia_mask=glia_mask)

        if self.learn_ignore_transitions and self.ignore_label is not None:
            output, mask = self.include_ignore_transitions(output, mask, labels)

        # Cast to be sure
        if not output.dtype == self.dtype:
            output = output.astype(self.dtype)
        #
        # print("affs: shape before binary", output.shape)
        if self.segmentation_to_binary:
            output = np.concatenate((self.to_binary_segmentation(labels)[None],
                                     output), axis=0)
        # print("affs: shape after binary", output.shape)

        # print("affs: shape before mask", output.shape)
        # We might want to carry the mask along.
        # If this is the case, we insert it after the targets.
        if self.retain_mask:
            mask = mask.astype(self.dtype, copy=False)
            if self.segmentation_to_binary:
                if self.ignore_label is None:
                    additional_mask = np.ones((1,) + labels.shape, dtype=self.dtype)
                else:
                    additional_mask = (labels[None] != self.ignore_label).astype(self.dtype)
                mask = np.concatenate([additional_mask, mask], axis=0)
            output = np.concatenate((output, mask), axis=0)
        # print("affs: shape after mask", output.shape)

        # We might want to carry the segmentation along for validation.
        # If this is the case, we insert it before the targets.
        if self.retain_segmentation:
            # Add a channel axis to labels to make it (C, Z, Y, X) before cating to output
            if self.retain_extra_masks:
                assert extra_masks is not None, "Extra masks where not passed and cannot be concatenated"
                output = np.concatenate((labels[None].astype(self.dtype, copy=False),
                                         extra_masks[None].astype(self.dtype, copy=False),
                                         output),
                                        axis=0)
            else:
                output = np.concatenate((labels[None].astype(self.dtype, copy=False), output),
                                    axis=0)

        if self.retain_glia_mask:
            assert self.glia_label is not None
            output = np.concatenate((output, np.expand_dims((extra_masks == self.glia_label).astype('float32'), axis=0)), axis=0)

        # print("affs: out shape", output.shape)
        return output

    def tensor_function(self, tensor):
        if tensor.ndim == 3:
            tensor = np.expand_dims(tensor, axis=0)
        output = self.input_function(tensor)
        return output


class Segmentation2AffinitiesDynamicOffsets(Segmentation2Affinities):
    def __init__(self, nb_offsets=1, max_offset_range=(1,30,30), min_offset_range=(0,0,0),
                 normalize_offsets=True, allowed_offsets=None,
                 **super_kwargs):
        raise DeprecationWarning("Update with glia mask")
        super(Segmentation2AffinitiesDynamicOffsets, self).__init__(offsets=[[1,1,1]], **super_kwargs)
        assert len(max_offset_range) == 3
        self.min_offset_range = min_offset_range
        self.max_offset_range = max_offset_range
        self.allowed_offsets = allowed_offsets

        assert nb_offsets == 1, "Not sure what the CNN should do with more than one..."
        self.nb_offsets = nb_offsets
        self.normalize_offsets = normalize_offsets


    def build_random_variables(self):
        if self.allowed_offsets is None:
            offsets = [[np.random.choice([-1, 1]) * np.random.randint(self.min_offset_range[i], self.max_offset_range[i]+1) for i in range(3)] for _ in range(self.nb_offsets)]
        else:
            offsets = [
                [np.random.choice([-1, 1]) * np.random.choice(self.allowed_offsets[i], size=1)
                 for i in range(3)] for _ in range(self.nb_offsets)]
        self.set_random_variable("offsets", offsets)

    def batch_function(self, batch):
        assert len(batch) % 2 == 0, "Assuming to have equal number of inputs and targets!"
        nb_inputs = int(len(batch) / 2)

        assert batch[nb_inputs].ndim == 3
        self.build_random_variables()
        random_offset = self.get_random_variable('offsets')
        affinities = self.dyn_input_function(batch[nb_inputs], random_offset)

        # Concatenate offsets at the end:
        if self.normalize_offsets:
            normalized_offsets = (np.array(random_offset) / np.array(self.max_offset_range)).flatten()
        else:
            normalized_offsets = np.array(random_offset).flatten().astype('float32')

        repeated_offsets = np.rollaxis(np.tile(normalized_offsets, reps=affinities.shape[1:] + (1,)), axis=-1, start=0)

        return batch[:nb_inputs] + (repeated_offsets, affinities)


    def dyn_input_function(self, tensor, offsets):
        # FIXME: is there a bettter way to avoid rewriting this code?
        # print("affs: in shape", tensor.shape)
        if self.ignore_label is not None:
            # output.shape = (C, Z, Y, X)
            output, mask = compute_affinities(tensor, offsets,
                                              ignore_label=self.ignore_label,
                                              have_ignore_label=True)
        else:
            output, mask = compute_affinities(tensor, offsets)

        # FIXME what does this do, need to refactor !
        # hack for platyneris data
        platy_hack = False
        if platy_hack:
            chan_mask = mask[1].astype('bool')
            output[0][chan_mask] = np.min(output[:2], axis=0)[chan_mask]

            chan_mask = mask[2].astype('bool')
            output[0][chan_mask] = np.minimum(output[0], output[2])[chan_mask]

        # Cast to be sure
        if not output.dtype == self.dtype:
            output = output.astype(self.dtype)
        #
        # print("affs: shape before binary", output.shape)
        if self.segmentation_to_binary:
            output = np.concatenate((self.to_binary_segmentation(tensor)[None],
                                     output), axis=0)
        # print("affs: shape after binary", output.shape)

        # print("affs: shape before mask", output.shape)
        # We might want to carry the mask along.
        # If this is the case, we insert it after the targets.
        if self.retain_mask:
            mask = mask.astype(self.dtype, copy=False)
            if self.segmentation_to_binary:
                mask = np.concatenate(((tensor[None] != self.ignore_label).astype(self.dtype), mask),
                                      axis=0)
            output = np.concatenate((output, mask), axis=0)
        # print("affs: shape after mask", output.shape)

        # We might want to carry the segmentation along for validation.
        # If this is the case, we insert it before the targets.
        if self.retain_segmentation:
            # Add a channel axis to tensor to make it (C, Z, Y, X) before cating to output
            output = np.concatenate((tensor[None].astype(self.dtype, copy=False), output),
                                    axis=0)

        # print("affs: out shape", output.shape)
        return output
