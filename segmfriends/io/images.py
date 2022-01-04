import numpy as np
import cv2
import imageio
import os.path

def write_segm_to_file(path, array):
    filename, extension = os.path.splitext(os.path.split(path)[1])
    if extension == ".png":
        imageio.imwrite(path, array.astype(np.uint16))
    elif extension == ".tif" or extension == ".tiff":
        cv2.imwrite(path, array.astype(np.uint16))
    else:
        raise ValueError("Only png and tif extensions supported")


def write_image_to_file(path, array):
    """
    TODO: to be modified and finetuned...
    """
    filename, extension = os.path.splitext(os.path.split(path)[1])
    if extension == ".png":
        imageio.imwrite(path, array)
    elif extension == ".tif" or extension == ".tiff":
        cv2.imwrite(path, array)
    else:
        raise ValueError("Only png and tif extensions supported")
