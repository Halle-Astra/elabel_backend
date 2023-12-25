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

from utils import base64_to_ndarray, ndarray2base64

app = Flask(__name__, static_folder='static')
app.debug = True
app.secret_key = os.urandom(24)
CORS(app, resources='/*', origins='*', methods=['GET', 'POST'])

import flask
print(flask.__version__)

from model_apps.sam import SAM
sam = SAM(device='cuda')


@app.route('/label.online/sam/upload',methods=["POST"])
@cross_origin()
def upload():
    # print("正在上传")
    # print(request.values['neg_points[1][]'])
    # print(request.json)
    # print(type(request.json.get('pos_points'))) # list type
    data_js = request.json
    img_b64 = data_js['image']
    pos_points = data_js.get('pos_points')
    neg_points = data_js.get('neg_points')
    img = base64_to_ndarray(img_b64)
    # print(img.shape) # 510, 384, 4 四通道图像

    points  = pos_points + neg_points
    labels = [1]*len(pos_points)+ [0]*len(neg_points)
    points = np.asarray(points)
    labels = np.asarray(labels)
    img = img[...,:-1]
    sam.set_image(img)
    mask = sam.predict(points, labels) # (h,w) # 另外，png的编码和jpg的编码结果不是一样的格式！
    mask = mask.astype(np.uint8)
    mask = mask*255

    mask_png = ndarray2base64(mask, '.png')
    mask_png = "data:image/png;base64,"+mask_png
    # print(mask.shape)
    return mask_png



def set_user_data(value):
    session.data = value
    return value

if __name__ == '__main__':
    app.run(host='0.0.0.0')
