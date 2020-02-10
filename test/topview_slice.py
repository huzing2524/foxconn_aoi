import cv2
import numpy as np
import sys
import os

cap = cv2.VideoCapture('/home/dsd/Desktop/foxconn-aoi/01_topview_moving.mov')
# cap.set(3, 1600)
# cap.set(4, 1200)


def main():
    filename = 131
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # cv2.imwrite(os.path.join('/home/dsd/Desktop/foxconn-aoi/test/dataset/video_frame', str(filename) + '.png'), frame)

            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.imshow('frame', frame)
            cv2.waitKey(0)
            filename += 1
        else:
            break


        # image_src = cv2.imread(sys.argv[1])  # pick.py my.png
        # if image_src is None:
        #     print("no image given!" )
        #     return

        # mask needs the hsv image
        # image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # # initial mask covering the background
        # upper = np.array([180, 49, 280])
        # lower = np.array([-10, -80, 75])
        # image_mask = cv2.inRange(image_hsv, lower, upper)
        # res = cv2.bitwise_and(frame, frame, mask=255 - image_mask)
        #
        # x = 52
        # width = 60
        # slices = []
        # name = 30
        # print(frame.shape)
        #
        # while x + width < frame.shape[0]:
        #     region = cv2.copyMakeBorder(frame[0:1300, x:x+width], 0, 0, 10, 10,
        #                                 cv2.BORDER_CONSTANT, value=[10, 10, 10])
        #     slices.append(region)
        #     x = x + width
        #     name += 1
        #     cv2.namedWindow('region', cv2.WINDOW_NORMAL)
        #     cv2.imshow('region', region)
        #     cv2.waitKey(0)
        #     # cv2.imwrite('/home/dsd/Desktop/foxconn-aoi/test/dataset/' + '%s' % name + '.png', region)
        #
        # # stiching it together
        # imstack = slices[0]  # np.array((image_src.shape[0], image_src.shape[1], 3))
        #
        # for slice in slices:
        #     imstack = np.hstack((imstack, slice))
        #
        # cv2.namedWindow('stack', cv2.WINDOW_NORMAL)
        # cv2.imshow('stack', imstack)
        # cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
