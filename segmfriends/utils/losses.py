import numpy as np


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
