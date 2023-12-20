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

from utils import base64_to_ndarray

app = Flask(__name__, static_folder='static')
app.debug = True
app.secret_key = os.urandom(24)
CORS(app, resources='/*', origins='*', methods=['GET', 'POST'])

import flask

print(flask.__version__)

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
    print(img.shape) # 510, 384, 4 四通道图像

    return "0"



def set_user_data(value):
    session.data = value
    return value

if __name__ == '__main__':
    app.run(host='0.0.0.0')
