# -*- coding:UTF-8 -*-

# @author  : admin
# @file    : images_cutting.py
# @datetime: 2023/2/14 10:00
# @software: PyCharm

"""
文件说明：
    
"""


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
