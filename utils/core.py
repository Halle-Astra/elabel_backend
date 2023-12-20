import base64
import os
from PIL import Image
import cv2
import numpy as np
import io
import os
import json
import requests as rq
from lxml import etree
import random
import time
import shutil

##### flask相关
# from flask import current_app as session
# from flask import render_template, request
# from flask import Flask
# from flask import current_app as session
# from flask import render_template, request
# from flask_cors import CORS, cross_origin




def ndarray2base64(img):
    stream = cv2.imencode('.jpg', img)[1]
    base64_str = str(base64.b64encode(stream))[2:-1]
    return base64_str


def bytes2ndarray(img):
    img_byte = bytearray(img.read())
    img = cv2.imdecode(np.asarray(img_byte, dtype='uint8'), cv2.IMREAD_UNCHANGED)  # cv2.imdecode应该不是必须的
    return img


def base64_to_ndarray(img):
    try:
        img = img.split(",")[1]  # 避免出现png格式的，比如裁剪后的直接toDataURL的结果就是png格式的，但是懒得修改了，改这里顺便实验一下一个图片有多少个等号
    except Exception as e:
        print(img[:100], e)
    img = base64.b64decode(img)  # 从print来看，img已经是base64的字符串了，所以不需要这行解码了，反而解码以后的东西转不成uint8了
    img = io.BytesIO(img)
    img = Image.open(img)
    image_np = np.asarray(img, dtype=np.uint8)  # 此时得到的是RGB图像
    return image_np
