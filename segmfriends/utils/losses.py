import numpy as np
import torch
import numbers


try:
    from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss
    from inferno.io.transform import Transform
except ImportError:
    raise ImportError("Couldn't find 'inferno' module, losses are not available")


class LabelTargetSorensenDiceLoss2D(SorensenDiceLoss):
    def __init__(self, last_class_is_ignore_class=False, *super_args, **super_kwargs):
        """

        :param index_ignore_class: If True, it indicates that the last class (or the labels with highest number in targets)
                    should be used as ignore-mask. In this case, predictions should have one class predicted less compared
                    to the number of labels in the target tensor.
        :param super_args:
        :param super_kwargs:
        """
        super(LabelTargetSorensenDiceLoss2D, self).__init__(*super_args, **super_kwargs)
        self.last_class_is_ignore_class = last_class_is_ignore_class


    def forward(self, prediction, target):
        """
        :param prediction:  
        :param target: 
        :return: 
        """

        nb_classes = prediction.shape[1]
        assert target.dim() == prediction.dim()
        shape = target.shape

        if self.last_class_is_ignore_class is not None:
            # Create an ignore mask:
            ignore_mask = target
            pass

        # Convert target to one-hot:
        target = target.view(-1, 1)

        target_onehot = torch.zeros_like(prediction)
        # TODO: generalize to 3D
        target_onehot = target_onehot.permute(0, 2, 3, 1).reshape(-1, nb_classes)

        # Scatter:
        target_onehot = target_onehot.scatter_(1, target, 1)

        # Reshape back:
        # TODO: generalize to 3D
        target_onehot = target_onehot.reshape(shape[0], shape[2], shape[3], nb_classes).permute(0,3,1,2)
        # print(target_onehot.shape)

        # # FIXME: hack to quickly invert background prediction
        # target_onehot[:,0] = 1 - target_onehot[:,0]
        # # I also need to invert prediction, otherwise softmax does not work
        # input[:,0] = 1 - input[:,0]

        return super(LabelTargetSorensenDiceLoss2D, self).forward(prediction, target_onehot)


from speedrun.log_anywhere import log_image


class MaskIgnoreClass(Transform):
    def __init__(self, **super_kwargs):
        """
        By default, ignore label is assumed to be the highest label in the targets
        (To be generalized, target labels have to be remapped continously)
        """
        super(MaskIgnoreClass, self).__init__(**super_kwargs)
        # assert isinstance(index_ignore_class, numbers.Integral)
        # self.index_ignore_class = index_ignore_class

    # for all batch requests, we assume that
    # we are passed prediction and target in `tensors`
    def batch_function(self, tensors):
        # Assert tensors and ignore class index:
        assert len(tensors) == 2
        prediction, target = tensors
        nb_classes = prediction.shape[1]
        index_ignore_class = int(nb_classes)
        # assert int(self.index_ignore_class) == int(nb_classes), "Ignore label should have index nb_pred_cl"

        log_image("target_before_remapping", target)

        # Find foreground mask:
        foreground_mask = target.clone().ne_(float(index_ignore_class))

        # Step 1:  Modify the targets and assign ignore label to label zero
        target[torch.eq(foreground_mask, 0)] = 0

        # Step 2: Modify the prediction tensor (maintaining gradient propagation):
        mask_tensor = foreground_mask.float().expand_as(prediction)
        mask_tensor.requires_grad = False
        # First, we zero out all predictions associated to ignored pixels:
        masked_prediction = prediction * mask_tensor
        # Then, we set predictions for ignored pixels and class 0 to 1.0 (to match targets and get no loss):
        mask_tensor = 1. - mask_tensor
        mask_tensor[:,1:] = 0.
        masked_prediction = masked_prediction + mask_tensor

        # log_image("prediction_after_mod", masked_prediction)

        return masked_prediction, target


class RemoveChannelDimension(Transform):
    def tensor_function(self, tensor):
        assert tensor.shape[1] == 1
        return tensor[:,0]

def get_loss_weights(samples_per_class, beta=0.9999):
    """
    Taken from
    https://towardsdatascience.com/handling-class-imbalanced-data-using-a-loss-specifically-made-for-it-6e58fd65ffab
    The output weights are normalized

    :param samples_per_class:
    :param beta: as one of 0.9, 0.99, 0.999, 0.9999. The highest the value, the more different the resulting weights
    :return:
    """
    nb_classes = samples_per_class.shape[0]
    effective_num = 1.0 - np.power(beta, samples_per_class.astype("float64"))
    # print(effective_num)
    weights = (1.0 - beta) / np.array(effective_num).astype("float64")
    weights = weights / np.sum(weights) * float(nb_classes)
    return weights
