import numpy as np


def convert_to_one_hot(labels):
    one_hot = np.zeros((labels.max() + 1,) + labels.shape, dtype='uint8')
    indices = np.indices(labels.shape)
    one_hot[labels.astype('int'), indices[0], indices[1], indices[2]] = 1
    return one_hot


def compute_IoU_numpy(predictions, targets):
    """
    The shape of both `predictions` and `targets` should be (nb_classes, z_shape, x_size_image, y_size_image)
    """

    nb_classes = predictions.shape[0]
    predictions = (predictions > 0.5).reshape(nb_classes, -1)
    targets = (targets > 0.5).reshape(nb_classes, -1)  # (nb_classes, batch * x * y)

    # Intersection: both GT and predictions are True (AND operator &)
    # Union: at least one of the two is True (OR operator |)
    IoU = 0
    IoU_per_class = []
    for cl in range(nb_classes):
        union = np.logical_or(predictions[cl], targets[cl]).sum()
        if union != 0:
            partial = np.logical_and(predictions[cl], targets[cl]).sum().astype('float32') / union.sum().astype(
                'float32')
        else:
            partial = 1.
        # print(cl, partial)
        IoU_per_class.append(partial)
        IoU = IoU + partial
    IoU = IoU / nb_classes

    return IoU, IoU_per_class


def compute_entropy(predictions):
    entropy = np.zeros(predictions.shape[1:])
    for cl in range(predictions.shape[0]):
        entropy -= predictions[cl] * np.log(predictions[cl])
    return entropy
