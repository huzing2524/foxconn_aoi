# -*- coding: utf-8 -*-
# @Time   : 19-7-5 下午2:30
# @Author : huziying
# @File   : match.py

import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10  # 设置最低匹配数量为10
# img1 = cv2.imread("area_5.png", 0)  # 读取第一个图像（小图像）
# img2 = cv2.imread("01_topview.png", 0)  # 读取第二个图像（大图像）

img1 = cv2.imread("pineapple_1.jpg", 0)  # 读取第一个图像（小图像）
img2 = cv2.imread("pineapple_2.jpg", 0)  # 读取第二个图像（大图像）

sift = cv2.xfeatures2d.SIFT_create()  # 创建sift检测器
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# 创建设置FLAAN匹配
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
mathces = flann.knnMatch(des1, des2, k=2)
good = []
# 过滤不合格的匹配结果，大于0.7的都舍弃
for m, n in mathces:
    if m.distance < 0.7 * n.distance:
        good.append(m)
# 如果匹配结果大于10，则获取关键点的坐标，用于计算变换矩阵
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 计算变换矩阵和掩膜
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
    matchesMask = mask.ravel().tolist()
    # 根据变换矩阵进行计算，找到小图像在大图像中的位置
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    cv2.polylines(img2, [np.int32(dst)], True, 0, 5, cv2.LINE_AA)
else:
    print(" Not Enough matches are found")
    matchesMask = None
# 画出特征匹配线
draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None,
                   matchesMask=matchesMask, flags=2)
# plt展示最终的结果
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
plt.imshow(img3), plt.show()
