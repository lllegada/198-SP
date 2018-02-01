import glob
import cv2
import numpy as np 	
import math
from matplotlib import pyplot as plt
from sklearn import svm, datasets
import os
import skimage
from skimage import io
from PIL import Image



img = cv2.imread("pen 10.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
thresh = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

cv2.imshow('thresh', thresh)
# thresh = cv2.imread("otsu.jpg",0)

size = np.size(thresh)
skel = np.zeros(thresh.shape,np.uint8)
			# threshold the image, setting all foreground pixels to
			# 255 and all background pixels to 0
ret, thresh = cv2.threshold(thresh,127,255,0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
done = False
# kernel = np.ones((5,5),np.uint8)
# close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel,iterations=2)
# open = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel)
# dilate = cv2.dilate(open,kernel)



# print("size: ",size)
# print(cv2.countNonZero(gray))
while(not done):	
	eroded = cv2.erode(thresh, element)
	temp = cv2.dilate(eroded, element)
	temp = cv2.subtract(thresh, temp)
	skel = cv2.bitwise_or(skel,temp)
	thresh = eroded.copy()

	zeros = size - cv2.countNonZero(thresh)

	print("\nsize: ", size)
	print("countNonZero: ", cv2.countNonZero(thresh))
	print("zeros", zeros)
	# break
	if zeros == size:
		done = True


# cv2.imshow("open", dilate)

# ===================================================
height, width = skel.shape
print(height, width)

# counts total number of 1s in the image
foreground = cv2.countNonZero(skel)
print("foreground px: ", foreground)

# find the number of 1s in each column
nonZeroYCount = np.count_nonzero(skel, axis = 0)
print("Numpy # of pixels: ", nonZeroYCount)
print("Num of col:\n", len(nonZeroYCount))


skel = cv2.cvtColor(skel, cv2.COLOR_GRAY2RGB)

# gets the index of columns which contains 0 or 1 values that might be PSCs
PSC_values = [i for i,val in enumerate(nonZeroYCount) if (val <= 1)] 

# draw line on PSC through the index from PSC_values
for x in PSC_values:
	point1 = (x, 0)
	point2 = (x, height)
	cv2.line(skel, point1, point2, (0, 0, 255), 1)
	print("non zero y count: ", nonZeroYCount[x])
	print("point1: ", point1)
	print("point2: ", point2)

print("PSC_values: ", PSC_values)
cv2.imshow("skeleton", skel)
cv2.waitKey(0)
cv2.destroyAllWindows()
