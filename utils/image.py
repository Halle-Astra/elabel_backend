# 专门的图像处理部分

import cv2

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
