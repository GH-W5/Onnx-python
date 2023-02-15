# -*- coding:UTF-8 -*-

# @author  : admin
# @file    : images_cutting.py
# @datetime: 2023/2/14 10:00
# @software: PyCharm

"""
文件说明：
    
"""
import numpy as np


def image_slide_cutting(width, height, window_size, overlapping_pixels=0, mode=0):
    """
    大影像滑动窗口切割生成器，返回当前切割得图片左上角得像素坐标(x,y)和实际切割的图像的尺寸(window_size_x, window_size_y)
    :param width: 原始大图像的宽
    :param height: 原始大图像的高
    :param window_size: 切割的图片尺寸大小
    :param overlapping_pixels: 窗口滑动时，重叠的像素大小。
    :param mode: 为0时，扫描到边界时，向前重叠保留切割尺寸不变，为其他值时，则不向前重叠，已实际剩余尺寸作为切割窗口大小
    :return:
    """
    if isinstance(window_size, int):
        window_size_x = window_size_y = window_size
    else:
        assert isinstance(window_size, (list, tuple))
        window_size_x, window_size_y = window_size[:2]
    # 存在切割尺寸大于原图尺寸的情况，特殊处理++
    window_size_y = min(window_size_y, height)
    window_size_x = min(window_size_x, width)
    stride_x = window_size_x - overlapping_pixels
    stride_y = window_size_y - overlapping_pixels
    # 按照从左往右，从上往下，逐行扫描
    if mode == 0:  # 最后一行或最后一列时，切割尺寸保持不变，图片切割或扫描推理时使用
        for y in (list(range(0, height - window_size_y, stride_y)) + [height - window_size_y]):
            for x in list(range(0, width - window_size_x, stride_x)) + [width - window_size_x]:
                yield (x, y), (window_size_x, window_size_y)

    else:  # 最后一行或列时，以实际剩余尺寸为准，大影像分块转矢量时使用
        assert overlapping_pixels == 0, 'mode值非0情况下，不需要重叠！'
        for y in (list_y := list(range(0, height, stride_y))):
            for x in (list_x := list(range(0, width, stride_x))):
                real_window_size_x, real_window_size_y = window_size_x, window_size_y
                if y == list_y[-1]:
                    real_window_size_y = height - y
                if x == list_x[-1]:
                    real_window_size_x = width - x
                yield (x, y), (real_window_size_x, real_window_size_y)


def crop_img(src_img, box, long_edge_pad_ratio=0.4, short_edge_pad_ratio=0.2):
    """Crop text region with their bounding box.

    Args:
        src_img (np.array): The original image.
        box (list[float | int]): Points of quadrangle.
        long_edge_pad_ratio (float): Box pad ratio for long edge
            corresponding to font size.
        short_edge_pad_ratio (float): Box pad ratio for short edge
            corresponding to font size.
    """
    assert len(box) == 8
    assert 0. <= long_edge_pad_ratio < 1.0
    assert 0. <= short_edge_pad_ratio < 1.0

    h, w = src_img.RasterYSize, src_img.RasterXSize
    points_x = np.clip(np.array(box[0::2]), 0, w)
    points_y = np.clip(np.array(box[1::2]), 0, h)

    box_width = np.max(points_x) - np.min(points_x)
    box_height = np.max(points_y) - np.min(points_y)
    font_size = min(box_height, box_width)

    if box_height < box_width:
        horizontal_pad = long_edge_pad_ratio * font_size
        vertical_pad = short_edge_pad_ratio * font_size
    else:
        horizontal_pad = short_edge_pad_ratio * font_size
        vertical_pad = long_edge_pad_ratio * font_size

    left = np.clip(int(np.min(points_x) - horizontal_pad), 0, w)
    top = np.clip(int(np.min(points_y) - vertical_pad), 0, h)
    right = np.clip(int(np.max(points_x) + horizontal_pad), 0, w)
    bottom = np.clip(int(np.max(points_y) + vertical_pad), 0, h)

    # dst_img = src_img[top:bottom, left:right]
    dst_img = src_img.ReadAsArray(int(left), int(top), int(box_width), int(box_height))

    return dst_img
