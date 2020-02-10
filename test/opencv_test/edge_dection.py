from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to the input image")
# ap.add_argument("-w", "--width", type=float, required=True,
#                 help="width of the left-most object in the image (in inches)")
# args = vars(ap.parse_args())

# 读取输入图片
# image = cv2.imread("/home/dsd/Desktop/foxconn-aoi/01_topview.png")
# # image = cv2.imread("/home/dsd/Desktop/foxconn-aoi/glass_1.jpg")
image = cv2.imread("/home/dsd/Desktop/foxconn-aoi/1.jpg")

# 输入图片灰度化
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 对灰度图片执行高斯滤波
# gray = cv2.GaussianBlur(gray, (7, 7), 0)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# 对滤波结果做边缘检测获取目标
edged = cv2.Canny(gray, 50, 100)
# 使用膨胀和腐蚀操作进行闭合对象边缘之间的间隙
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow("Image", edged)
cv2.waitKey(0)

# 在边缘图像中寻找物体轮廓（即物体）
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# 对轮廓按照从左到右进行排序处理
(cnts, _) = contours.sort_contours(cnts)
# 初始化 'pixels per metric'
pixelsPerMetric = None

# 循环遍历每一个轮廓
for c in cnts:
    # 如果当前轮廓的面积太少，认为可能是噪声，直接忽略掉
    if cv2.contourArea(c) < 100:
        continue

    # 根据物体轮廓计算出外切矩形框
    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # 按照top-left, top-right, bottom-right, bottom-left的顺序对轮廓点进行排序，并绘制外切的BB，用绿色的线来表示
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    # 绘制BB的4个顶点，用红色的小圆圈来表示
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    # 分别计算top-left 和top-right的中心点和bottom-left 和bottom-right的中心点坐标
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # 分别计算top-left和top-right的中心点和top-righ和bottom-right的中心点坐标
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # 绘制BB的4条边的中心点，用蓝色的小圆圈来表示
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # 在中心点之间绘制直线，用紫红色的线来表示
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
             (255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
             (255, 0, 255), 2)

    # 计算两个中心点之间的欧氏距离，即图片距离
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # 初始化测量指标值，参考物体在图片中的宽度已经通过欧氏距离计算得到，参考物体的实际大小已知
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / 0.955

    # 计算目标的实际大小（宽和高），用英尺来表示
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    # 在图片中绘制结果
    cv2.putText(orig, "{:.1f}in".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}in".format(dimB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)

    # 显示结果
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Image", orig)
    cv2.waitKey(0)
