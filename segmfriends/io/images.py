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

def read_uint8_img(img_path, add_all_channels_if_needed=True):
    # TODO: rename and move to io module together with function exporting segmentation file
    assert os.path.isfile(img_path), "Image {} not found".format(img_path)

    extension = os.path.splitext(img_path)[1]
    if extension == ".tif" or extension == ".tiff":
        # img = cv2.imread(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # print(img.dtype, img.min(), img.max())
        # Sometimes some images are loaded in float and cannot be automatically converted to uint8:
        # FIXME: check type and then convert to uint8 (or uint16??)
        # if img.dtype != 'uint8':
        #     print("Warning, image {} has type {} and it was converted to uint8".format(os.path.split(img_path)[1], img.dtype))
        if img.dtype == 'uint16':
            img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
        assert img.dtype == 'uint8'
            # # img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
            # img = img - img.min()
            # img = (img / img.max() * 255.).astype('uint8')
    elif extension == ".png":
        img = imageio.imread(img_path)
    else:
        raise ValueError("Extension {} not supported".format(extension))
    if len(img.shape) == 2 and add_all_channels_if_needed:
        # Add channel dimension:
        img = np.stack([img for _ in range(3)])
        img = np.rollaxis(img, axis=0, start=3)
    # assert len(img.shape) == 3 and img.shape[2] == 3, img.shape

    return img
