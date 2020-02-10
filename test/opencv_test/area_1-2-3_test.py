import numpy as np
import cv2
import imutils

img1 = cv2.imread('./area_5.png')
# img1 = cv2.imread('/home/dsd/Desktop/foxconn-aoi/test/area_2.png')
# img1 = cv2.imread('/home/dsd/Desktop/foxconn-aoi/test/area_3.png')
# img1 = cv2.imread('/home/dsd/Desktop/foxconn-aoi/test/area_5.png')
H, W, C = img1.shape  # 106, 175, 3
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1_blurred = cv2.GaussianBlur(img1_gray, (5, 5), 0)
img1_thresh = cv2.threshold(img1_blurred, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
img1_edges = cv2.Canny(img1_thresh, 200, 400, 3)
orb = cv2.ORB_create(50000, scoreType=cv2.ORB_FAST_SCORE, nlevels=20, edgeThreshold=5)
kp1, des1 = orb.detectAndCompute(img1_edges, None)
img1_keys = cv2.drawKeypoints(img1_edges, kp1, outImage=np.array([]), color=(0, 255, 0), flags=0)
cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
cv2.imshow("img1", img1_keys)
cv2.waitKey(0)

img2 = cv2.imread('../01_topview.png')
# img2 = cv2.imread('/home/dsd/Desktop/foxconn-aoi/01_topview_sample.png')
h, w, c = img2.shape  # 2048 2448 3
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2_blurred = cv2.GaussianBlur(img2_gray, (5, 5), 0)
img2_thresh = cv2.threshold(img2_blurred, 190, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
img2_edges = cv2.Canny(img2_thresh, 200, 400, 3)
kp2, des2 = orb.detectAndCompute(img2_edges, None)
cnts = cv2.findContours(img2_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)


def getMatches(part):
    kp2, des2 = orb.detectAndCompute(part, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    if matches is not None:
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        pts = np.float32([[0, 0], [0, H - 1], [W - 1, H - 1], [W - 1, 0]]).reshape(-1, 1, 2)
        if M is not None and len(matchesMask) > 0:
            dst = cv2.perspectiveTransform(pts, M)
            img2 = cv2.polylines(part, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)
            img3 = cv2.drawMatches(img1, kp1, part, kp2, good, None, **draw_params)
            cv2.imshow("homography", img3)
            return dst
    else:
        # print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    return None


for x in range(0, w - W, int(W)):  # step = 0.05 * W, it will cost a lot of time
    print(x)
    sw = img2_edges[0:int(h / 3), x:x + int(W)].copy()

    result = cv2.matchTemplate(img2_edges, img1_edges, cv2.TM_CCOEFF_NORMED)

    # cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
    # cv2.imshow("img2", img2)
    # cv2.waitKey(1)

    threshold = 0.1
    loc = np.where(result >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img2, (pt[0] + x, pt[1]), (pt[0] + x + W, pt[1] + H), (0, 0, 255), 2)

    # cv2.namedWindow('cropped', cv2.WINDOW_NORMAL)
    # cv2.imshow("cropped", sw)
    # cv2.waitKey(1)

    boxes = None  # getMatches(sw)

    if boxes is not None:
        # Draw lines between the corners (the mapped object in the scene)
        # print(boxes)
        boxes[0][0][0] += x
        boxes[1][0][0] += x
        boxes[2][0][0] += x
        boxes[3][0][0] += x
        print(boxes)
        cv2.polylines(img2, np.int32([boxes]), True, (0, 255, 0), 4)

cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
cv2.imshow("img2", img2)
cv2.waitKey(0)
