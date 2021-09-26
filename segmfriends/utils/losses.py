import numpy as np
import torch

try:
    from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss
except ImportError:
    raise ImportError("Couldn't find 'inferno' module, losses are not available")


class LabelTargetSorensenDiceLoss2D(SorensenDiceLoss):
    def forward(self, input, target):
        assert target.dim() == input.dim()
        shape = target.shape
        nb_classes = input.shape[1]

        # Convert target to one-hot:
        target.view(-1, 1)

        target_onehot = torch.zeros_like(input)
        # TODO: generalize to 3D
        target_onehot.permute(0, 2, 3, 1).view(-1, nb_classes)

        # Scatter:
        target_onehot.scatter_(1, target, 1)

        # Reshape back:
        # TODO: generalize to 3D
        target_onehot.view(shape[0], shape[2], shape[3], nb_classes).permute(0,3,1,2)
        print(target_onehot.shape)

        return super(LabelTargetSorensenDiceLoss2D, self).forward(input, target_onehot)



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
