import torch
import numpy as np
import onnxruntime

rd = np.random.RandomState(888)


def ort_ocr_det():
    matrix1 = rd.random((1, 3, 736, 736)).astype(np.float32)

    ort_inputs = {'input': matrix1}
    ort_session = onnxruntime.InferenceSession(
        'E:/datasets/ocr/kqrs_train/train_data/DbNet_r18/end2end.onnx',
        providers=['CPUExecutionProvider']
    )  # notice
    ort_output = ort_session.run(['output'], ort_inputs)[0]
    pass


def ort_ocr_seg():
    matrix1 = rd.random((1, 3, 48, 256)).astype(np.float32)

    ort_inputs = {'input': matrix1}
    ort_session = onnxruntime.InferenceSession(
        'E:/datasets/ocr/kqrs_train/train_data/Sar_r31//end2end.onnx',
        providers=['CUDAExecutionProvider']
    )  # notice

    ort_output = ort_session.run(['output'], ort_inputs)[0] #
    pass


if __name__ == '__main__':
    # ort_ocr_det()
    ort_ocr_seg()
