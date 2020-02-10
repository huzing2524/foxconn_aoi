import numpy as np
import cv2
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


show = None

MIN_MATCH_COUNT = 10

tmpl = cv2.imread('01_a1.png')
H, W, C = tmpl.shape
tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
tmpl_blurred = cv2.GaussianBlur(tmpl_gray, (5, 5), 0)
# tmpl1 was 120,255
# 01_a1 is 2,255 but somehow the first value seems not to be that important
ret, tmp_thresh = cv2.threshold(tmpl_blurred, 2, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
tmp = cv2.Canny(tmp_thresh, 200, 400, 3)
# orb = cv2.ORB_create(10000, scoreType=cv2.ORB_FAST_SCORE, nlevels=8, edgeThreshold = 5)
orb = cv2.ORB_create(50000, scoreType=cv2.ORB_FAST_SCORE, nlevels=20, edgeThreshold=5)
kp1, des1 = orb.detectAndCompute(tmp, None)
if des1 is None:
    print("des1 is None, meaning nothing found!")
tmp_keys = cv2.drawKeypoints(tmp, kp1, outImage=np.array([]), color=(0, 255, 0), flags=0)


def getMatches(part):
    kp2, des2 = orb.detectAndCompute(part, None)
    if des2 is None:
        print("des2 is None, meaning nothing found!")
        return None
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
    bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1, des2, k=2)
    # matches = flann.knnMatch(des1,des2,k=2)
    matches = bf.knnMatch(np.asarray(des1, np.uint8), np.asarray(des2, np.uint8), k=2)

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
            cv2.imshow("homography", img3)
            return dst
    else:
        # print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    return None


# manual settings
alpha = 0.2
beta = 2.3

while (True):
    frame = cv2.imread("01_topview.png")

    # Our operations on the frame come here
    # alpha = 1.3
    # beta = 40
    # sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    sobelx = frame
    # find minimum and maximum intensities
    minVal = np.amin(sobelx)
    maxVal = np.amax(sobelx)
    # default values
    # alpha = 255.0/(maxVal - minVal)
    # beta = -minVal * 255.0/(maxVal - minVal)
    # to debug the alpha and beta values print(minVal, maxVal, alpha, beta)
    cgray = cv2.convertScaleAbs(sobelx, alpha, beta)

    gray = cv2.cvtColor(cgray, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]
    thresh2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 20)
    ret, thresh = cv2.threshold(blurred, 190, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 200, 400, 3)

    # find contours in the edge map
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) == 0:
        continue

    h, w, c = frame.shape
    width = w - W + 1
    height = h - H + 1

    # find template in sliding window
    # use template x and move by 10% each slice
    for x in range(0, w - W, int(0.05 * W)):
        # print("slice %d:%d %d:%d" % (0,h-1,x,x+int(2*W)-1, ))
        sw = edges[0:int(h / 3), x:x + int(W)].copy()

        # TM_CCOEFF_NORMED
        # TM_SQDIFF
        result = cv2.matchTemplate(sw, tmp, cv2.TM_CCOEFF_NORMED)
        # print(result)
        threshold = 0.18
        loc = np.where(result >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, (pt[0] + x, pt[1]), (pt[0] + x + W, pt[1] + H), (0, 0, 255), 2)

        # cv2.imshow("cropped", sw)
        # file = "crop_" + str(x) + ".png"
        # cv2.imwrite(file, sw)
        # print("slice %d" % (x))
        boxes = getMatches(sw)
        if boxes is not None:
            # Draw lines between the corners (the mapped object in the scene)
            # print(boxes)
            boxes[0][0][0] += x
            boxes[1][0][0] += x
            boxes[2][0][0] += x
            boxes[3][0][0] += x
            # print(boxes)
            cv2.polylines(frame, np.int32([boxes]), True, (0, 255, 0), 4)
            # cv2.line(frame, boxes[0], boxes[1], (0, 255, 0), 4)
            # cv2.line(frame, boxes[1], boxes[2], (0, 255, 0), 4)
            # cv2.line(frame, boxes[2], boxes[3], (0, 255, 0), 4)
            # cv2.line(frame, boxes[3], boxes[0], (0, 255, 0), 4)
    cv2.imshow("intermediate", frame)

    # sort the contours from left-to-right and, then initialize the
    # distance colors and reference object
    (cnts, _) = contours.sort_contours(cnts)
    colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),
              (255, 0, 255))
    refObj = None

    if show is None:
        show = frame

    # loop over the contours individually
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

    # cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 0), 2)
    # cv2.drawContours(frame, [refObj[0].astype("int")], -1, (0, 255, 0), 2)
    # cv2.drawContours(frame, [c.astype("int")], -1, (0, 255, 0), 2)

    # Display the resulting frame
    wshow = cv2.resize(show, (1200, 800))
    cv2.imshow('frame', wshow)
    cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('o'):
        show = frame
    elif key & 0xFF == ord('e'):
        show = edges
    elif key & 0xFF == ord('b'):
        show = blurred
    elif key & 0xFF == ord('t'):
        show = thresh
    elif key & 0xFF == ord('l'):
        show = tmp
    elif key & 0xFF == ord('k'):
        show = tmp_keys
    elif key & 0xFF == ord('v'):
        show = tmpl_gray
    elif key & 0xFF == ord('c'):
        show = tmp_thresh
    elif key & 0xFF == ord('a'):
        alpha = alpha + 0.1
    elif key & 0xFF == ord('y'):
        alpha = alpha - 0.1
    elif key & 0xFF == ord('i'):
        beta = beta + 0.1
    elif key & 0xFF == ord('j'):
        beta = beta - 0.1
    elif key & 0xFF == ord('q'):
        break

# When everything done, release the capture
cv2.destroyAllWindows()
