import base64
import os

from flask import Flask
from flask import render_template, request
from PIL import Image
import cv2
import numpy as np
import io
from flask import current_app as session
import os
from flask_cors import CORS, cross_origin
import json
import requests as rq
from lxml import etree
import random
import time
import shutil
from flask import current_app as session
from flask import render_template, request
app = Flask(__name__, static_folder='static')
app.debug = True
app.secret_key = os.urandom(24)
CORS(app, resources='/*', origins='*', methods=['GET', 'POST'])

import flask

print(flask.__version__)

@app.route('/label.online/upload',methods=["POST"])
@cross_origin()
def upload():
    print("正在上传")
    print(request.values)
    return "0"




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




def set_user_data(value):
    session.data = value
    return value






def storageImgs_preprocessing(img,resize_side=512):
    """主要是将图像居中和缩放，统一图像并减少传输的数据量"""
    h,w,c = img.shape
    square_side = min(h,w)
    center_x = w//2
    center_y = h//2
    left = center_x - square_side//2
    right = center_x+square_side//2
    top = center_y - square_side//2
    bottom = center_y + square_side//2
    img = img[top:bottom, left:right]
    img = cv2.resize(img, (resize_side,resize_side))
    return img



if __name__ == '__main__':
    app.run(host='0.0.0.0')
