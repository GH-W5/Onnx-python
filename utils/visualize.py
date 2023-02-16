# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import shutil
import urllib
import warnings

import cv2
import numpy as np
# import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def overlay_mask_img(img, mask):
    """Draw mask boundaries on image for visualization.

    Args:
        img (ndarray): The input image.
        mask (ndarray): The instance mask.

    Returns:
        img (ndarray): The output image with instance boundaries on it.
    """
    assert isinstance(img, np.ndarray)
    assert isinstance(mask, np.ndarray)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    return img


def show_feature(features, names, to_uint8, out_file=None):
    """Visualize a list of feature maps.

    Args:
        features (list(ndarray)): The feature map list.
        names (list(str)): The visualized title list.
        to_uint8 (list(1|0)): The list indicating whether to convent
            feature maps to uint8.
        out_file (str): The output file name. If set to None,
            the output image will be shown without saving.
    """

    num = len(features)
    row = col = math.ceil(math.sqrt(num))

    for i, (f, n) in enumerate(zip(features, names)):
        plt.subplot(row, col, i + 1)
        plt.title(n)
        if to_uint8[i]:
            f = f.astype(np.uint8)
        plt.imshow(f)
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)


def show_img_boundary(img, boundary):
    """Show image and instance boundaires.

    Args:
        img (ndarray): The input image.
        boundary (list[float or int]): The input boundary.
    """
    assert isinstance(img, np.ndarray)

    cv2.polylines(
        img, [np.array(boundary).astype(np.int32).reshape(-1, 1, 2)],
        True,
        color=(0, 255, 0),
        thickness=1)
    plt.imshow(img)
    plt.show()


def tile_image(images):
    """Combined multiple images to one vertically.

    Args:
        images (list[np.ndarray]): Images to be combined.
    """
    assert isinstance(images, list)
    assert len(images) > 0

    for i, _ in enumerate(images):
        if len(images[i].shape) == 2:
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_GRAY2BGR)

    widths = [img.shape[1] for img in images]
    heights = [img.shape[0] for img in images]
    h, w = sum(heights), max(widths)
    vis_img = np.zeros((h, w, 3), dtype=np.uint8)

    offset_y = 0
    for image in images:
        img_h, img_w = image.shape[:2]
        vis_img[offset_y:(offset_y + img_h), 0:img_w, :] = image
        offset_y += img_h

    return vis_img


def imshow_text_label(img,
                      pred_label,
                      gt_label,
                      show=False,
                      win_name='',
                      wait_time=-1,
                      out_file=None):
    """Draw predicted texts and ground truth texts on images.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        pred_label (str): Predicted texts.
        gt_label (str): Ground truth texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str): The filename of the output.
    """
    assert isinstance(img, (np.ndarray, str))
    assert isinstance(pred_label, str)
    assert isinstance(gt_label, str)
    assert isinstance(show, bool)
    assert isinstance(win_name, str)
    assert isinstance(wait_time, int)

    img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_COLOR)

    src_h, src_w = img.shape[:2]
    resize_height = 64
    resize_width = int(1.0 * src_w / src_h * resize_height)
    img = cv2.resize(img, (resize_width, resize_height))
    h, w = img.shape[:2]

    if is_contain_chinese(pred_label):
        pred_img = draw_texts_by_pil(img, [pred_label], None)
    else:
        pred_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        cv2.putText(pred_img, pred_label, (5, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)
    images = [pred_img, img]

    if gt_label != '':
        if is_contain_chinese(gt_label):
            gt_img = draw_texts_by_pil(img, [gt_label], None)
        else:
            gt_img = np.ones((h, w, 3), dtype=np.uint8) * 255
            cv2.putText(gt_img, gt_label, (5, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 0, 0), 2)
        images.append(gt_img)

    img = tile_image(images)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        cv2.imwrite(img, out_file)

    return img


def gen_color():
    """Generate BGR color schemes."""
    color_list = [(101, 67, 254), (154, 157, 252), (173, 205, 249),
                  (123, 151, 138), (187, 200, 178), (148, 137, 69),
                  (169, 200, 200), (155, 175, 131), (154, 194, 182),
                  (178, 190, 137), (140, 211, 222), (83, 156, 222)]
    return color_list


def draw_polygons(img, polys):
    """Draw polygons on image.

    Args:
        img (np.ndarray): The original image.
        polys (list[list[float]]): Detected polygons.
    Return:
        out_img (np.ndarray): Visualized image.
    """
    dst_img = img.copy()
    color_list = gen_color()
    out_img = dst_img
    for idx, poly in enumerate(polys):
        poly = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(
            img,
            np.array([poly]),
            -1,
            color_list[idx % len(color_list)],
            thickness=cv2.FILLED)
        out_img = cv2.addWeighted(dst_img, 0.5, img, 0.5, 0)
    return out_img


def get_optimal_font_scale(text, width):
    """Get optimal font scale for cv2.putText.

    Args:
        text (str): Text in one box.
        width (int): The box width.
    """
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(
            text,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=scale / 10,
            thickness=1)
        new_width = textSize[0][0]
        if new_width <= width:
            return scale / 10
    return 1


def draw_texts(img, texts, boxes=None, draw_box=True, on_ori_img=False):
    """Draw boxes and texts on empty img.

    Args:
        img (np.ndarray): The original image.
        texts (list[str]): Recognized texts.
        boxes (list[list[float]]): Detected bounding boxes.
        draw_box (bool): Whether draw box or not. If False, draw text only.
        on_ori_img (bool): If True, draw box and text on input image,
            else, on a new empty image.
    Return:
        out_img (np.ndarray): Visualized image.
    """
    color_list = gen_color()
    h, w = img.shape[:2]
    if boxes is None:
        boxes = [[0, 0, w, 0, w, h, 0, h]]
    assert len(texts) == len(boxes)

    if on_ori_img:
        out_img = img
    else:
        out_img = np.ones((h, w, 3), dtype=np.uint8) * 255
    for idx, (box, text) in enumerate(zip(boxes, texts)):
        if draw_box:
            new_box = [[x, y] for x, y in zip(box[0::2], box[1::2])]
            Pts = np.array([new_box], np.int32)
            cv2.polylines(
                out_img, [Pts.reshape((-1, 1, 2))],
                True,
                color=color_list[idx % len(color_list)],
                thickness=1)
        min_x = int(min(box[0::2]))
        max_y = int(
            np.mean(np.array(box[1::2])) + 0.2 *
            (max(box[1::2]) - min(box[1::2])))
        font_scale = get_optimal_font_scale(
            text, int(max(box[0::2]) - min(box[0::2])))
        cv2.putText(out_img, text, (min_x, max_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), 1)

    return out_img


def draw_texts_by_pil(img,
                      texts,
                      boxes=None,
                      draw_box=True,
                      on_ori_img=False,
                      font_size=None,
                      fill_color=None,
                      draw_pos=None,
                      return_text_size=False):
    """Draw boxes and texts on empty image, especially for Chinese.

    Args:
        img (np.ndarray): The original image.
        texts (list[str]): Recognized texts.
        boxes (list[list[float]]): Detected bounding boxes.
        draw_box (bool): Whether draw box or not. If False, draw text only.
        on_ori_img (bool): If True, draw box and text on input image,
            else on a new empty image.
        font_size (int, optional): Size to create a font object for a font.
        fill_color (tuple(int), optional): Fill color for text.
        draw_pos (list[tuple(int)], optional): Start point to draw each text.
        return_text_size (bool): If True, return the list of text size.

    Returns:
        (np.ndarray, list[tuple]) or np.ndarray: Return a tuple
        ``(out_img, text_sizes)``, where ``out_img`` is the output image
        with texts drawn on it and ``text_sizes`` are the size of drawing
        texts. If ``return_text_size`` is False, only the output image will be
        returned.
    """

    color_list = gen_color()
    h, w = img.shape[:2]
    if boxes is None:
        boxes = [[0, 0, w, 0, w, h, 0, h]]
    if draw_pos is None:
        draw_pos = [None for _ in texts]
    assert len(boxes) == len(texts) == len(draw_pos)

    if fill_color is None:
        fill_color = (0, 0, 0)

    if on_ori_img:
        out_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        out_img = Image.new('RGB', (w, h), color=(255, 255, 255))
    out_draw = ImageDraw.Draw(out_img)

    text_sizes = []
    for idx, (box, text, ori_point) in enumerate(zip(boxes, texts, draw_pos)):
        if len(text) == 0:
            continue
        min_x, max_x = min(box[0::2]), max(box[0::2])
        min_y, max_y = min(box[1::2]), max(box[1::2])
        color = tuple(list(color_list[idx % len(color_list)])[::-1])
        if draw_box:
            out_draw.line(box, fill=color, width=1)
        dirname, _ = os.path.split(os.path.abspath(__file__))
        font_path = os.path.join(dirname, 'font.TTF')
        if not os.path.exists(font_path):
            url = ('https://download.openmmlab.com/mmocr/data/font.TTF')
            print(f'Downloading {url} ...')
            local_filename, _ = urllib.request.urlretrieve(url)
            shutil.move(local_filename, font_path)
        tmp_font_size = font_size
        if tmp_font_size is None:
            box_width = max(max_x - min_x, max_y - min_y)
            tmp_font_size = int(0.9 * box_width / len(text))
        fnt = ImageFont.truetype(font_path, tmp_font_size)
        if ori_point is None:
            ori_point = (min_x + 1, min_y + 1)
        out_draw.text(ori_point, text, font=fnt, fill=fill_color)
        text_sizes.append(fnt.getsize(text))

    del out_draw

    out_img = cv2.cvtColor(np.asarray(out_img), cv2.COLOR_RGB2BGR)

    if return_text_size:
        return out_img, text_sizes

    return out_img


def is_contain_chinese(check_str):
    """Check whether string contains Chinese or not.

    Args:
        check_str (str): String to be checked.

    Return True if contains Chinese, else False.
    """
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def det_recog_show_result(img, end2end_res, out_file=None):
    """Draw `result`(boxes and texts) on `img`.

    Args:
        img (str or np.ndarray): The image to be displayed.
        end2end_res (dict): Text detect and recognize results.
        out_file (str): Image path where the visualized image should be saved.
    Return:
        out_img (np.ndarray): Visualized image.
    """
    img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_COLOR)
    boxes, texts = [], []
    for res in end2end_res:
        boxes.append(res['box'])
        texts.append(res['text'])
    box_vis_img = draw_polygons(img, boxes)

    if is_contain_chinese(''.join(texts)):
        text_vis_img = draw_texts_by_pil(img, texts, boxes)
    else:
        text_vis_img = draw_texts(img, texts, boxes)

    h, w = img.shape[:2]
    out_img = np.ones((h, w * 2, 3), dtype=np.uint8)
    out_img[:, :w, :] = box_vis_img
    out_img[:, w:, :] = text_vis_img

    if out_file:
        cv2.imwrite(out_img, out_file)
    imshow(out_img, 'inference results')
    return out_img


def imshow(img, win_name: str = '', wait_time: int = 0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    # x, y = img.shape[0:2]
    # scale = 1024 / x
    # img_resize = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(win_name, img)
    if wait_time == 0:  # prevent from hanging if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)
