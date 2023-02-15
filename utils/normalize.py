# -*- coding:UTF-8 -*-

# @author  : admin
# @file    : normalize.py
# @datetime: 2023/2/14 16:52
# @software: PyCharm

"""
文件说明：
    
"""
import cv2
import numpy as np


def normalize_(img, mean, std, to_rgb=False):
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    img = img.copy().astype(np.float32)
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace

    img = img.transpose(2, 0, 1)
    return np.expand_dims(img, axis=0)
