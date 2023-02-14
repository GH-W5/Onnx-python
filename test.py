# -*- coding:UTF-8 -*-

# @author  : admin
# @file    : test.py
# @datetime: 2023/2/13 14:54
# @software: PyCharm

"""
文件说明：
    
"""
# import mmcv
#
# filename = 'F:/workspace/mmocr_api/data/王贵/3.png'
# file_client = mmcv.FileClient()
# img_bytes = file_client.get(filename)
# img = mmcv.imfrombytes(img_bytes)

import  cv2
import numpy as np

aa = np.random.random((736,736))
bb = cv2.resize(aa, (800,800))
pass
