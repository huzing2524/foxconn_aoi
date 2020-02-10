import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt


def match_template():
    """matchTemplate"""
    img1 = cv2.imread('/home/dsd/Desktop/foxconn-aoi/test/opencv_test/area_5.png')
    # img1 = cv2.imread('/home/dsd/Desktop/foxconn-aoi/test/area_2.png')
    # img1 = cv2.imread('/home/dsd/Desktop/foxconn-aoi/test/area_3.png')
    # img1 = cv2.imread('/home/dsd/Desktop/foxconn-aoi/test/area_5.png')
    H, W, C = img1.shape  # 106, 175, 3
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1_blurred = cv2.GaussianBlur(img1_gray, (5, 5), 0)
    img1_thresh = cv2.threshold(img1_blurred, 2, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img1_edges = cv2.Canny(img1_thresh, 200, 400, 3)
    orb = cv2.ORB_create(50000, scoreType=cv2.ORB_FAST_SCORE, nlevels=20, edgeThreshold=5)
    kp1, des1 = orb.detectAndCompute(img1_edges, None)
    img1_keys = cv2.drawKeypoints(img1_edges, kp1, outImage=np.array([]), color=(0, 255, 0), flags=0)
    cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
    cv2.imshow("img1", img1_edges)
    cv2.waitKey(0)

    # img2 = cv2.imread('/home/dsd/Desktop/foxconn-aoi/01_topview.png')
    img2 = cv2.imread('/home/dsd/Desktop/foxconn-aoi/01_topview_sharp_pins.png')
    h, w, c = img2.shape  # 2048 2448 3
    cgray = cv2.convertScaleAbs(img2, 0.2, 2.3)
    img2_gray = cv2.cvtColor(cgray, cv2.COLOR_BGR2GRAY)
    img2_blurred = cv2.GaussianBlur(img2_gray, (5, 5), 0)
    img2_thresh = cv2.threshold(img2_blurred, 190, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img2_edges = cv2.Canny(img2_thresh, 200, 400, 3)
    kp2, des2 = orb.detectAndCompute(img2_edges, None)
    img2_keys = cv2.drawKeypoints(img2_edges, kp2, outImage=np.array([]), color=(0, 255, 0), flags=0)
    cnts = cv2.findContours(img2_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
    cv2.imshow("img2", img2_edges)
    cv2.waitKey(0)

    result = cv2.matchTemplate(img2, img1, cv2.TM_CCOEFF_NORMED)
    threshold = 0.61
    loc = np.where(result >= threshold)
    for pt in zip(*loc[::-1]):
        # cv2.rectangle(img2, (pt[0], pt[1]), (pt[0] + W, pt[1] + H), (0, 0, 255), 2)
        cv2.rectangle(img2, pt, (pt[0] + W, pt[1] + H), (0, 0, 255), 2)

    cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    cv2.imshow("res", img2)
    cv2.waitKey(0)


def bf_match():
    """cv2.BFMatch"""
    img1 = cv2.imread('/home/dsd/Desktop/foxconn-aoi/test/opencv_test/area_5.png', 0)
    # img1 = cv2.imread('/home/dsd/Desktop/foxconn-aoi/test/opencv_test/area_5_backup.png', 0)
    # img1 = cv2.imread('/home/dsd/Desktop/foxconn-aoi/test/opencv_test/area_5_bf_matcher.png')
    img2 = cv2.imread('/home/dsd/Desktop/foxconn-aoi/01_topview_slice_01.png', 0)
    # orb = cv2.ORB_create()  # 建立orb特征检测器
    # orb = cv2.ORB_create(nfeatures=500,
    #                      scaleFactor=1.2,
    #                      nlevels=8,
    #                      edgeThreshold=31,
    #                      firstLevel=0,
    #                      WTA_K=2,
    #                      scoreType=cv2.ORB_HARRIS_SCORE,
    #                      patchSize=31,
    #                      fastThreshold=20)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)  # 计算img1中的特征点和描述符
    kp2, des2 = orb.detectAndCompute(img2, None)  # 计算img2中的
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # 建立匹配关系
    mathces = bf.match(des1, des2)  # 匹配描述符
    mathces = sorted(mathces, key=lambda x: x.distance)  # 据距离来排序
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, mathces[:40], None, flags=2)  # 画出匹配关系
    cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    cv2.imshow("res", img3)
    cv2.waitKey(0)


def flann():
    """cv.FlannBasedMatcher"""
    img1 = cv2.imread('/home/dsd/Desktop/foxconn-aoi/test/opencv_test/area_5.png', 0)
    img2 = cv2.imread('/home/dsd/Desktop/foxconn-aoi/01_topview_slice_01.png', 0)
    """
    AttributeError: module 'cv2.cv2' has no attribute 'xfeatures2d'
    pip3 install opencv-python==3.3.0.10 opencv-contrib-python==3.3.0.10
    """
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(des1, des2, k=2)
    matchesMask = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:  # 舍弃小于0.7的匹配结果
            matchesMask[i] = [1, 0]

    drawParams = dict(matchColor=(0, 0, 255), singlePointColor=(255, 0, 0), matchesMask=matchesMask,
                      flags=0)  # 给特征点和匹配的线定义颜色
    resultimage = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **drawParams)  # 画出匹配的结果
    cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    cv2.imshow("res", resultimage)
    cv2.waitKey(0)


def flann_2():
    """cv.FlannBasedMatcher"""
    MIN_MATCH_COUNT = 10  # 设置最低匹配数量为10
    img1 = cv2.imread("/home/dsd/Desktop/foxconn-aoi/test/opencv_test/area_5_bf_matcher.png", 0)  # 读取第一个图像（小图像）
    img2 = cv2.imread("/home/dsd/Desktop/foxconn-aoi/01_topview_sharp_pins.png", 0)  # 读取第二个图像（大图像）

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
    cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    cv2.imshow("res", img3)
    cv2.waitKey(0)


if __name__ == '__main__':
    # pass
    # match_template()
    # bf_match()
    # flann()
    flann_2()
