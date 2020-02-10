import cv2
import numpy as np
from matplotlib import pyplot as plt


img1 = cv2.imread('/home/dsd/Desktop/foxconn-aoi/test/area_5.png')
H, W, C = img1.shape
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1_blurred = cv2.GaussianBlur(img1_gray, (5, 5), 0)
img1_thresh = cv2.threshold(img1_blurred, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
edges1 = cv2.Canny(img1_thresh, 200, 400, 3)
orb = cv2.ORB_create(50000, scoreType=cv2.ORB_FAST_SCORE, nlevels=20, edgeThreshold=5)

kp1, des1 = orb.detectAndCompute(edges1, None)
tmp_keys = cv2.drawKeypoints(edges1, kp1, outImage=np.array([]), color=(0, 255, 0), flags=0)

img2 = cv2.imread('/home/dsd/Desktop/foxconn-aoi/01_topview.png')
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2_blurred = cv2.GaussianBlur(img2_gray, (5, 5), 0)
img2_thresh = cv2.threshold(img2_blurred, 190, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

edges2 = cv2.Canny(img2_thresh, 200, 400, 3)
kp2, des2 = orb.detectAndCompute(edges2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
if matches is not None:
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

# print(len(good))
# if len(good) > 10:
# src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
# dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#
# M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
# matchesMask = mask.ravel().tolist()
#
# pts = np.float32([[0, 0], [0, H - 1], [W - 1, H - 1], [W - 1, 0]]).reshape(-1, 1, 2)

# print(M), print(len(matchesMask))
# if M is not None and len(matchesMask) > 0:
# dst = cv2.perspectiveTransform(pts, M)
# img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
# draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
#                    singlePointColor=None,
#                    matchesMask=matchesMask,  # draw only inliers
#                    flags=2)
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None)

cv2.namedWindow('test', cv2.WINDOW_NORMAL)
cv2.imshow("test", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
