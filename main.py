import numpy as np
import matplotlib.pyplot as plt
import cv2

"""Read in image"""
# img = cv2.imread('../DATA/rainbow.jpg')
# plt.imshow(img)
# plt.show()

"""Read in image, in grayscale"""
# img = cv2.imread('../DATA/rainbow.jpg', 0)
# plt.imshow(img, cmap='gray')
# plt.show()

"""
the threshold esentially creates a range, anything inside or outside gets manipulated
"""

"""
# 127 is just roughly half of 255
# THRESH_BINARY_INV inverts all the results
# THRESH_TRUNC if over threshold, brings it down to the threshold, otherwise keep source
# THRESH_TOZERO is essentially the inverse of THRESH_TRUNC
"""

# ret,thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# print(ret)
# plt.imshow(thresh1,cmap='gray')
# plt.show()

img = cv2.imread('../DATA/crossword.jpg',0)
# plt.imshow(img,cmap='gray')
# plt.show()

def show_pic(img):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()

show_pic(img)

# this imperfect due to overlap of the colors
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
show_pic(th1)

"""
- which image to use
- the max value
- adaptive method
- threshold method
- block size - the number of pixels around a pixel, to compare (pixel neighborhood), needs to be odd, {3,5,11} are common values
- C value - constant value
"""
"""
ADAPTIVE_THRESH_GAUSSIAN
"""
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
show_pic(th2)

blended = cv2.addWeighted(src1=th1, alpha=0.6, src2=th2, beta=0.4, gamma=0)
show_pic(blended)