# -*- coding:utf-8 -*-
import os
import logging
import numpy as np
import cv2
import onnxruntime
from PIL import Image
from osgeo import gdal

from utils.images_cutting import image_slide_cutting
from utils.db_postprocessor import DBPostprocessor
from utils.visualize import imshow_pred_boundary

mean = np.array([123.675, 116.28, 103.53])
std = np.array([58.395, 57.12, 57.375])

Image.MAX_IMAGE_PIXELS = None


class OCRInference:

    def __init__(self, model_path, device='CPUExecutionProvider'):
        self.model_path = model_path
        self.model = onnxruntime.InferenceSession(self.model_path, providers=[device])
        # 文字定位模型后处理，获取预测的文本边界
        self.dbnet_post = DBPostprocessor(text_repr_type='quad')
        self.model_name = "OCRInference"
        self.logger = self.get_logger()
        self.logger.info('开始'.center(10, '*'))

    def __call__(self, img_dir):
        img_path_list = self.get_images_path(img_dir)

        for img_path in img_path_list:
            img = self.read_image(img_path)
            output_dir = os.path.splitext(img_path)[0] + '.json'
            self.sliding_window_detection(img,img_path, output_dir)

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

    def sliding_window_detection(self, image_data, img_path, output_path, window_size=736, stride=736):
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
        labels = [0] * len(boundaries)
        imshow_pred_boundary(img_path, boundaries, labels, 0.3, 'green', 'green',
                             win_name='onnxruntime')

    def predict_img(self, image):
        image = image.transpose(1, 2, 0)
        # 数据预处理
        origin_shape = image.shape
        image = cv2.resize(image, (736, 736))
        image = self.normalize_(image, mean, std)
        ort_inputs = {'input': image}
        preds = np.squeeze(self.model.run(['output'], ort_inputs)[0])
        prob_map = preds[0, :, :]
        out = cv2.resize(prob_map, (origin_shape[0], origin_shape[1]))
        return out

    def normalize_(self, img, mean, std, to_rgb=True):
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
        # if to_rgb:
        #     cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace

        img = img.transpose(2, 0, 1)
        return np.expand_dims(img, axis=0)

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
    model_path = 'E:/datasets/ocr/kqrs_train/train_data/DbNet_r18/end2end.onnx'
    img_dir = 'F:/workspace/mmocr_api/data/orc_test2.jpg'
    ocr_infer = OCRInference(model_path)
    ocr_infer(img_dir)
