import numpy as np
import cv2
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils


MIN_MATCH_COUNT = 10
show = None


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


tmpl = cv2.imread('/home/dsd/Desktop/foxconn-aoi/test/area_5.png')
H, W, C = tmpl.shape  # 106, 175, 3
tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
tmpl_blurred = cv2.GaussianBlur(tmpl_gray, (5, 5), 0)
tmp_thresh = cv2.threshold(tmpl_blurred, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # 阈值 120
tmp = cv2.Canny(tmp_thresh, 200, 400, 3)
# orb = cv2.ORB_create(10000, scoreType=cv2.ORB_FAST_SCORE, nlevels=8, edgeThreshold = 5)
orb = cv2.ORB_create(50000, scoreType=cv2.ORB_FAST_SCORE, nlevels=20, edgeThreshold=5)
kp1, des1 = orb.detectAndCompute(tmp, None)
tmp_keys = cv2.drawKeypoints(tmp, kp1, outImage=np.array([]), color=(0, 255, 0), flags=0)


def getMatches(part):
    kp2, des2 = orb.detectAndCompute(part, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    if matches is not None:
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

    if len(good) > MIN_MATCH_COUNT:
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
            img3 = cv2.drawMatches(tmp, kp1, part, kp2, good, None, **draw_params)
            cv2.namedWindow('homography', cv2.WINDOW_NORMAL)
            cv2.imshow("homography", img3)
            cv2.waitKey(0)
            return dst
    else:
        # print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    return None


frame = cv2.imread('/home/dsd/Desktop/foxconn-aoi/01_topview.png')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 190, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
edges = cv2.Canny(thresh, 200, 400, 3)

cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
cv2.imshow("edges", edges)

# find contours in the edge map
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

h, w, c = frame.shape  # 2048, 2448, 3
width = w - W + 1  # 2274 大图-小图
height = h - H + 1  # 1943

# box = getMatches(edges)
for x in range(0, w - W, int(0.05 * W)):
    # print("slice %d:%d %d:%d" % (0,h-1,x,x+int(2*W)-1, ))
    sw = edges[0:int(h / 3), x:x + int(W)].copy()

    cv2.namedWindow('cropped', cv2.WINDOW_NORMAL)
    cv2.imshow("cropped", sw)

    # TM_CCOEFF_NORMED
    # TM_SQDIFF
    result = cv2.matchTemplate(sw, tmp, cv2.TM_CCOEFF_NORMED)
    # print(result)
    threshold = 0.18
    loc = np.where(result >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, (pt[0] + x, pt[1]), (pt[0] + x + W, pt[1] + H), (0, 0, 255), 2)

    # file = "crop_" + str(x) + ".png"
    # cv2.imwrite(file, sw)
    # print("slice %d" % (x))

    boxes = None  # getMatches(sw)
    # print('boxes', boxes)
    if boxes is not None:
        # Draw lines between the corners (the mapped object in the scene)
        # print(boxes)
        boxes[0][0][0] += x
        boxes[1][0][0] += x
        boxes[2][0][0] += x
        boxes[3][0][0] += x
        print(boxes)
        cv2.polylines(frame, np.int32([boxes]), True, (0, 255, 0), 4)

cv2.namedWindow('intermediate', cv2.WINDOW_NORMAL)
cv2.imshow("intermediate", frame)
cv2.waitKey(0)

(cnts, _) = contours.sort_contours(cnts)
colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),
          (255, 0, 255))
refObj = None

if show is None:
    show = frame

for c in cnts:
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 100:
        continue

    # compute the rotated bounding box of the contour
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)

    # compute the center of the bounding box
    cX = np.average(box[:, 0])
    cY = np.average(box[:, 1])

    # if this is the first contour we are examining (i.e.,
    # the left-most contour), we presume this is the
    # reference object
    if refObj is None:
        # unpack the ordered bounding box, then compute the
        # midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-right and
        # bottom-right
        (tl, tr, br, bl) = box
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # compute the Euclidean distance between the midpoints,
        # then construct the reference object
        D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        # / args["width"]
        refObj = (box, (cX, cY), D / 1.0)
        continue

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.imshow('frame', show)
key = cv2.waitKey(0)

# if key & 0xFF == ord('o'):
#     show = frame
# elif key & 0xFF == ord('e'):
#     show = edges
# elif key & 0xFF == ord('b'):
#     show = blurred
# elif key & 0xFF == ord('t'):
#     show = thresh
# elif key & 0xFF == ord('l'):
#     show = tmp
# elif key & 0xFF == ord('k'):
#     show = tmp_keys
# elif key & 0xFF == ord('q'):
#     break

# cv2.destroyAllWindows()
