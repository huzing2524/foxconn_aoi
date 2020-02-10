# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())

# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread("01_topview.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# tmp = cv2.Canny(blurred,200,400,3)

output = blurred.copy()

# detect circles in the image
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20)
# , param1=20, param2=100, minRadius=20,maxRadius=80)

if circles is not None:
    print(len(circles))

# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    # show the output image
    cv2.imshow("output", output)
    # tmp = cv2.Canny(image,200,400,3)
    cv2.waitKey(0)
