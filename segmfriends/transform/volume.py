import numpy as np
from scipy.ndimage import zoom

try:
    from inferno.io.transform import Compose, Transform
except ImportError:
    raise ImportError("Volume transforms requires inferno")

from ..utils.various import parse_data_slice
from .filters import downscale, countless3d


class DuplicateGtDefectedSlices(Transform):
    def __init__(self, defects_label=2, ignore_label=0, **super_kwargs):
        self.defects_label = defects_label
        self.ignore_label = ignore_label
        super(DuplicateGtDefectedSlices, self).__init__(**super_kwargs)

    def batch_function(self, batch):
        assert len(batch) == 2, "This transform expects two tensors: raw, gt_segm"
        targets = batch[1]

        if targets.ndim == 4 and targets.shape[0] != 1:
            assert targets.shape[0] == 2, "Targets can have max two channels: gt labels and extra masks"

            defect_mask = targets[1] == self.defects_label

            # On the first slice we should never have defects:
            if defect_mask[0].max():
                print("WARNING: defects on first slice!!!")
                # In this special case, we set GT of the first slice to ignore label:
                targets[:, 0] = self.ignore_label

            # For each defect, get GT from the previous slice
            for z_indx in range(1, defect_mask.shape[0]):
                # Copy GT:
                targets[0, z_indx][defect_mask[z_indx]] = targets[0, z_indx-1][defect_mask[z_indx]]
                # Copy masks:
                targets[1, z_indx][defect_mask[z_indx]] = targets[1, z_indx-1][defect_mask[z_indx]]
        else:
            assert targets.ndim == 3, "Targets should be either 3 or 4 dimensional tensor"

        return (batch[0], targets)


class MergeExtraMasks(Transform):
    """
    Merge extra_masks loaded from file (with masks of boundary pixels, glia segments and defected slices in the dataset)
    with the mask of slices modified during augmentation (which now contain artifacts, are black out, etc...)
    """
    def __init__(self, defects_label=2, **super_kwargs):
        self.defects_label = defects_label
        super(MergeExtraMasks, self).__init__(**super_kwargs)

    def batch_function(self, batch):
        extra_masks = None
        if len(batch) == 3:
            raw_data, gt, extra_masks = batch
        else:
            assert len(batch) == 2, "This transform requires either two or three tensors: raw, GT_segm, (extra_masks)"
            raw_data, gt = batch
        if isinstance(raw_data, (tuple, list)):
            assert len(raw_data) == 2, "Raw data should be either one tensor or a list of two tensors"
            # Create an empty tensor, if None:
            extra_masks = np.zeros_like(gt) if extra_masks is None else extra_masks
            # Combine defects (in the original data and from augmentation):
            raw_data, augmented_defected_mask = raw_data
            extra_masks[augmented_defected_mask.astype('bool')] = self.defects_label

        if extra_masks is not None:
            # Concatenate labels in one tensor:
            gt = np.stack([gt,extra_masks])

        return (raw_data, gt)


class ReplicateTensorsInBatch(Transform):
    """
    Creates replicates of the tensors in the batch (used to create multi-scale inputs).
    """
    def __init__(self,
                 indices_to_replicate,
                 **super_kwargs):
        super(ReplicateTensorsInBatch, self).__init__(**super_kwargs)
        self.indices_to_replicate = indices_to_replicate

    def batch_function(self, batch):
        new_batch = []
        for indx in self.indices_to_replicate:
            assert indx < len(batch)
            new_batch.append(np.copy(batch[indx]))
        return new_batch


class From3Dto2Dtensors(Transform):
    """
    Tensors are expected to be 4D (channel dim + spatial ones) or 3D (only spatial dimensions). Return 2D by checking that 3rd dimension can be compressed.
    """
    def __init__(self,
                 **super_kwargs):
        super(From3Dto2Dtensors, self).__init__(**super_kwargs)

    def batch_function(self, batch):
        new_batch = [None for _ in range(len(batch))]
        for indx in range(len(batch)):
            if batch[indx].ndim == 4:
                assert batch[indx].shape[1] == 1
                new_batch[indx] = batch[indx][:,0]
            else:
                assert batch[indx].ndim == 3
                assert batch[indx].shape[0] == 1
                new_batch[indx] = batch[indx][0]
        return new_batch



class DownSampleAndCropTensorsInBatch(Transform):
    def __init__(self,
                 ds_factor=(1, 2, 2),
                 crop_factor=None,
                 crop_slice=None,
                 order=None,
                 use_countless=False,
                 **super_kwargs):
        """
        :param ds_factor: If factor is 2, then downscaled to half-resolution
        :param crop_factor: If factor is 2, the central crop of half-size is taken
        :param crop_slice: Alternatively, a string can be passed representing the desired crop
        :param order: downscaling order. By default, it is set to 3 for 'float32'/'float64' and to 0 for
                    integer/boolean data types.
        """
        super(DownSampleAndCropTensorsInBatch, self).__init__(**super_kwargs)
        self.order = order
        self.ds_factor = ds_factor
        if crop_factor is not None:
            assert isinstance(crop_factor, (tuple, list))
            # assert crop_slice is None
        if crop_slice is not None:
            # assert crop_slice is not None
            assert isinstance(crop_slice, str)
            crop_slice = parse_data_slice(crop_slice)
        self.crop_factor = crop_factor
        self.crop_slice = crop_slice
        self.use_countless = use_countless

    def volume_function(self, volume):
        # Crop:
        if self.crop_factor is not None:
            shape = volume.shape
            cropped_shape = [int(shp / crp_fct) for shp, crp_fct in zip(shape, self.crop_factor)]
            offsets = [int((shp - crp_shp) / 2) for shp, crp_shp in zip(shape, cropped_shape)]
            crop_slc = tuple(slice(off, off + crp_shp) for off, crp_shp in zip(offsets, cropped_shape))
            volume = volume[crop_slc]

        # Crop using the passed crop_slice:
        if self.crop_slice is not None:
            volume = volume[self.crop_slice]

        # Downscale the volume:
        if (np.array(self.ds_factor) != 1).any():
            if self.order is None:
                if volume.dtype in [np.dtype('float32'), np.dtype('float64')]:
                    volume = downscale(volume, self.ds_factor, filter="box")
                elif volume.dtype in [np.dtype('int8'), np.dtype('int16'), np.dtype('int32'), np.dtype('int64')]:
                    ndim = volume.ndim
                    ds_factor = [self.ds_factor for _ in range(ndim)] if isinstance(self.ds_factor, int) else \
                        self.ds_factor
                    if self.use_countless:
                        assert ds_factor == [2,2,2], "Only 3d countless supported for the moment"
                        volume = countless3d(volume)
                    else:
                        ds_slice = tuple(slice(None, None, ds_factor[d]) for d in range(ndim))
                        volume = volume[ds_slice]
                else:
                    raise ValueError

        return volume

    def apply_to_torch_tensor(self, tensor):
        assert tensor.ndimension() == 5
        assert (np.array(self.ds_factor) == 1).all(), "Atm not implemented (Zoom not applicable to tensors)"

        # Crop:
        cropped = tensor
        if self.crop_factor is not None:
            shape = tensor.shape[-3:]
            cropped_shape = [int(shp/crp_fct) for shp, crp_fct in zip(shape, self.crop_factor)]
            offsets = [int((shp-crp_shp)/2) for shp, crp_shp in zip(shape, cropped_shape)]
            crop_slc = (slice(None), slice(None)) + tuple(slice(off, off+crp_shp) for off, crp_shp in zip(offsets, cropped_shape))
            cropped = tensor[crop_slc]

        if self.crop_slice is not None:
            # Crop using the passed crop_slice:
            cropped = cropped[(slice(None), slice(None)) + self.crop_slice]

        return cropped


class CheckBatchAndChannelDim(Transform):
    def __init__(self, dimensionality, *super_args, **super_kwargs):
        super(CheckBatchAndChannelDim, self).__init__(*super_args, **super_kwargs)
        self.dimensionality = dimensionality

    def batch_function(self, batch):
        output_batch = []
        for tensor in batch:
            if tensor.ndimension() == self.dimensionality:
                output_batch.append(tensor.unsqueeze(0).unsqueeze(0))
            elif tensor.ndimension() == self.dimensionality + 1:
                output_batch.append(tensor.unsqueeze(0))
            elif tensor.ndimension() == self.dimensionality + 2:
                output_batch.append(tensor)
            else:
                raise ValueError
        return tuple(output_batch)


class ApplyAndRemoveMask(Transform):
    def __init__(self, first_invert_target=False,
                 first_invert_prediction=False,
                 **super_kwargs):
        super(ApplyAndRemoveMask, self).__init__(**super_kwargs)
        self.first_invert_target = first_invert_target
        self.first_invert_prediction = first_invert_prediction

    def batch_function(self, tensors):
        assert len(tensors) == 2
        prediction, target = tensors

        # validate the prediction
        assert prediction.dim() in [4, 5], prediction.dim()
        assert target.dim() == prediction.dim(), "%i, %i" % (target.dim(), prediction.dim())
        assert target.size(1) == 2 * prediction.size(1), "%i, %i" % (target.size(1), prediction.size(1))
        assert target.shape[2:] == prediction.shape[2:], "%s, %s" % (str(target.shape), str(prediction.shape))
        seperating_channel = target.size(1) // 2
        mask = target[:, seperating_channel:]
        target = target[:, :seperating_channel]
        mask.requires_grad = False

        if self.first_invert_prediction:
            prediction = 1. - prediction
        if self.first_invert_target:
            target = 1. - target

        # mask prediction and target with mask
        prediction = prediction * mask
        target = target * mask
        return prediction, target
