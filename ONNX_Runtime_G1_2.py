# -*- coding:utf-8 -*-
import json
import math
import os
import logging

import numpy as np
import cv2
import onnxruntime
from PIL import Image
from osgeo import gdal

from utils.normalize import normalize_
from utils.fileio import list_from_file
from utils.images_cutting import image_slide_cutting
from utils.db_postprocessor import DBPostprocessor

# import mmcv
# from utils.visualize import det_recog_show_result

Image.MAX_IMAGE_PIXELS = None


class OCRSeg:
    '''
    文字识别onnx推理代码
    '''
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    def __init__(self, seg_model_path, key, device='CPUExecutionProvider'):
        self.seg_model_path = seg_model_path
        self.model = onnxruntime.InferenceSession(self.seg_model_path, providers=[device])
        self.key = key
        self.padding_idx = None
        self.idx2char = []
        for line_num, line in enumerate(list_from_file(self.key)):
            line = line.strip('\r\n')
            if len(line) > 1:
                raise ValueError('Expect each line has 0 or 1 character, '
                                 f'got {len(line)} characters '
                                 f'at line {line_num + 1}')
            if line != '':
                self.idx2char.append(line)
        start_end_token = '<BOS/EOS>'
        unknown_token = '<UKN>'
        self.idx2char.append(unknown_token)
        self.idx2char.append(start_end_token)
        self.end_idx = len(self.idx2char) - 1

    def __call__(self, image):
        image = image.transpose(1, 2, 0)[..., ::-1]
        # mmcv.imshow(image, 'inference results')
        # 数据预处理
        image = self.resize(image)
        image = normalize_(image, self.mean, self.std)
        ort_inputs = {'input': image}
        pred = self.model.run(['output'], ort_inputs)[0]
        label_indexes, label_scores = self.tensor2idx(pred)
        label_strings = self.idx2str(label_indexes)
        # flatten batch results
        results = []
        for string, score in zip(label_strings, label_scores):
            results.append(dict(text=string, score=score))

        return results[0]

    def resize(self, img):
        dst_height = 48
        dst_min_width = 48
        dst_max_width = 256

        ori_height, ori_width = img.shape[:2]

        new_width = math.ceil(float(dst_height) / ori_height * ori_width)
        width_divisor = 4
        # make sure new_width is an integral multiple of width_divisor.
        if new_width % width_divisor != 0:
            new_width = round(new_width / width_divisor) * width_divisor
        new_width = max(dst_min_width, new_width)
        resize_width = min(dst_max_width, new_width)
        img_resize = cv2.resize(img, (resize_width, dst_height), )
        if new_width < dst_max_width:
            img_resize = cv2.copyMakeBorder(
                img_resize,
                0,
                max(dst_height - img_resize.shape[0], 0),
                0,
                max(dst_max_width - img_resize.shape[1], 0),
                0,
                value=0)
        return img_resize / 255

    def tensor2idx(self, outputs):
        """
        Convert output tensor to text-index
        Args:
            outputs (tensor): model outputs with size: N * T * C
            img_metas (list[dict]): Each dict contains one image info.
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]]
            scores (list[list[float]]): [[0.9,0.8,0.95,0.97,0.94],
                                         [0.9,0.9,0.98,0.97,0.96]]
        """
        batch_size = outputs.shape[0]
        ignore_indexes = [self.padding_idx]
        indexes, scores = [], []
        for idx in range(batch_size):
            seq = outputs[idx, :, :]
            max_value = np.max(seq, axis=1)
            max_idx = np.argmax(seq, axis=1)
            str_index, str_score = [], []
            output_index = max_idx.tolist()
            output_score = max_value.tolist()
            for char_index, char_score in zip(output_index, output_score):
                if char_index in ignore_indexes:
                    continue
                if char_index == self.end_idx:
                    break
                str_index.append(char_index)
                str_score.append(char_score)

            indexes.append(str_index)
            scores.append(str_score)

        return indexes, scores

    def idx2str(self, indexes):
        """Convert indexes to text strings.

        Args:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
        Returns:
            strings (list[str]): ['hello', 'world'].
        """
        assert isinstance(indexes, list)

        strings = []
        for index in indexes:
            string = [self.idx2char[i] for i in index]
            strings.append(''.join(string))

        return strings


class OCRInference:
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])

    def __init__(self, det_model_path, seg_model_path, keys, device='CPUExecutionProvider'):
        self.det_model = onnxruntime.InferenceSession(det_model_path, providers=[device])
        self.ocr_seg_infer = OCRSeg(seg_model_path, keys)
        # 文字定位模型后处理，获取预测的文本边界
        self.dbnet_post = DBPostprocessor(text_repr_type='quad')
        self.model_name = "OCRInference"
        self.logger = self.get_logger()
        self.logger.info('开始'.center(50, '='))

    def __call__(self, img_dir):
        img_path_list = self.get_images_path(img_dir)

        for img_path in img_path_list:
            img = self.read_image(img_path)
            output_dir = os.path.splitext(img_path)[0] + '.json'
            self.sliding_window_detection(img, img_path, output_dir)

    def get_images_path(self, img_dir):
        if os.path.isdir(img_dir):  # 目录
            imgs_name_list = os.listdir(img_dir)
            imgs_path_list = [os.path.join(img_dir, img_name) for img_name in imgs_name_list]
            return imgs_path_list
        else:  # 单个文件或者多个文件逗号拼接
            return img_dir.split(",")

    def read_image(self, img_path):
        """
        读取原始大图像，返回图像数组 d array
        :param img_path:
        :return:
        """
        img = None
        try:
            img = gdal.Open(img_path)
            self.geo_transform = img.GetGeoTransform()
            self.projection_ref = img.GetProjectionRef()
            self.logger.info(f'{img_path} 图片读取成功,图片尺寸：{img.RasterYSize, img.RasterXSize, img.RasterCount}')
        except Exception as e:
            self.logger.info(f'{img_path} 图片无法读取,错误信息：{str(e)}')

        return img

    def sliding_window_detection(self, image_data, img_path, output_path, window_size=736):
        """
        滑动窗口对原始图像进行剪裁，并检测，将预测结果保存到指定文件夹
        :param image_array:
        :param window_size:
        :param output_dir:
        :return:
        """
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        height, width = image_data.RasterYSize, image_data.RasterXSize
        mask_crop = np.zeros([height, width], dtype=np.float32)

        self.logger.info(f'顺序扫描检测检测中。。。')
        # 按照从左往右，从上往下，逐行扫描
        for (x, y), (window_size_x, window_size_y) in image_slide_cutting(width, height, window_size):
            img_cropped = image_data.ReadAsArray(x, y, window_size_x, window_size_y)
            img_pred = self.predict_img(img_cropped)
            mask_crop[y:y + window_size_y, x:x + window_size_x] = img_pred  # 拼接到mask
        boundaries = self.dbnet_post(mask_crop)
        # 可视化
        # labels = [0] * len(boundaries)
        # imshow_pred_boundary(img_path, boundaries, labels, 0.3, 'green', 'green',
        #                      win_name='onnxruntime')
        img_e2e_res = []
        for bbox in boundaries:
            box_res = {}
            box_res['box'] = [round(x) for x in bbox[:-1]]
            box_res['box_score'] = float(bbox[-1])
            min_x = min(bbox[0:-1:2])
            min_y = min(bbox[1:-1:2])
            max_x = max(bbox[0:-1:2])
            max_y = max(bbox[1:-1:2])
            box_img = image_data.ReadAsArray(int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
            recog_result = self.ocr_seg_infer(box_img)
            text = recog_result['text']
            text_score = recog_result['score']
            if isinstance(text_score, list):
                text_score = sum(text_score) / max(1, len(text))
            box_res['text'] = text
            box_res['text_score'] = text_score
            img_e2e_res.append(box_res)

        # det_recog_show_result(img_path, img_e2e_res)
        self.export_json(img_e2e_res, output_path)

    def export_json(self, img_e2e_res, export):
        all_res = []
        for res in img_e2e_res:
            print(res)
            weight = res['text_score']
            if weight > 1.0: weight = 1.0
            all_res.append({
                "pos": (np.asarray(res['box']).reshape([4, 2])).tolist(),
                "value": res['text'],
                "weight": weight
            })
        res_dic = {"result": all_res}
        json_str = json.dumps(res_dic, indent=4, ensure_ascii=False)
        with open(export, 'w', encoding='utf-8') as json_file:
            json_file.write(json_str)

    def predict_img(self, image):
        image = image.transpose(1, 2, 0)
        # 数据预处理
        image = normalize_(image, self.mean, self.std)
        ort_inputs = {'input': image}
        preds = np.squeeze(self.det_model.run(['output'], ort_inputs)[0])
        prob_map = preds[0, :, :]
        return prob_map

    def get_logger(self):
        """
        创建日志对象
        :return:
        """
        os.makedirs('logs', exist_ok=True)
        logger = logging.getLogger(__name__)
        if not logger.handlers:  # 如果已经存在相应进程，就直接返回，避免重复打印
            fh = logging.FileHandler('./logs/' + self.model_name + '_log.log', mode='w', encoding='utf-8')
            ch = logging.StreamHandler()
            logger.addHandler(fh)
            logger.addHandler(ch)
            format = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
            fh.setFormatter(format)
            logger.setLevel(logging.INFO)
        return logger


if __name__ == '__main__':
    det_model_path = 'E:/datasets/ocr/kqrs_train/train_data/DbNet_r18/end2end.onnx'
    seg_model_path = 'E:/datasets/ocr/kqrs_train/train_data/Sar_r31/end2end.onnx'
    keys = 'E:/datasets/ocr/kqrs_train/train_data/Sar_r31/keys.txt'
    img_dir = 'F:/workspace/mmocr_api/data/orc_test2.jpg'
    seg_img_path = 'F:/workspace/mmocr_api/data/王贵/2.png'

    ocr_infer = OCRInference(det_model_path, seg_model_path, keys)
    ocr_infer(img_dir)

    # ocr_seg_infer = OCRSeg(seg_model_path, keys)
    # img = gdal.Open(seg_img_path)
    # print(ocr_seg_infer(img.ReadAsArray()))
