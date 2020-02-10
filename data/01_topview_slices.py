import cv2
import numpy as np

def main():
    import sys

    image_src = cv2.imread(sys.argv[1])  # pick.py my.png
    if image_src is None:
        print ("no image given!")
        return

    # mask needs the hsv image
    image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
    # initial mask covering the background
    upper =  np.array([180, 49, 280])
    lower =  np.array([-10, -80, 75])
    image_mask = cv2.inRange(image_hsv,lower,upper)
    res = cv2.bitwise_and(image_src,image_src,mask = 255 - image_mask)

    # slicing the image
    x = 52
    width = 60
    slices = []
    print(image_src.shape)
    while x+width < image_src.shape[1]:
        region = cv2.copyMakeBorder(image_src[0:image_src.shape[0],x-34:x+width],0,0,10,10,cv2.BORDER_CONSTANT,value=[10,10,10])
        slices.append(region)
        x = x + width

    # stiching it together
    imstack = slices[0] # np.array((image_src.shape[0], image_src.shape[1], 3))

    for slice in slices:
        imstack = np.hstack((imstack,slice))

    cv2.imshow('stack',imstack)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()