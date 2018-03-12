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


# """remove_blobs function aims to remove repeatedly the blobs from the binary image"""
# Initially preprocessed image using OTSU_ method still contains background noises. In order to remove it,
# Get contours,check for noises(area < 10), color those noises with black
# 
def remove_blobs(image):
	# Get contours of binary image. 
	_, thresh1_contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	for c in thresh1_contours:
		x, y, w, h = cv2.boundingRect(c)
		area = w * h 
		if area > 10:
			pass
		else:
			image[y:y+h,x:x+w] = np.zeros((h,w),np.uint8)
	
	return image
	
# """ init_remove_blobs function"""
#  function called for removing blob in binary image
#  draws rectangles on contours of final colored image
def init_remove_blobs(image):
	image = remove_blobs(image)
	
	# Get the contours of the "after-remove-blobs" image output 
	_, final_contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	# Convert to RGB color scale in order to draw colored rectangle
	image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

	# Draw rectangles on contours
	for c in final_contours:
		x, y, w, h = cv2.boundingRect(c)
		cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 0, 255), 1)
	
	# Return the image with 3 channels,image with binary channel 
	return image_rgb,image



	
img = cv2.imread("Pen_pics/pen 2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
thresh = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# Used to show the original image's noise
cv2.imshow("OLD_THRESH",thresh)
size = np.size(thresh)
skel = np.zeros(thresh.shape,np.uint8)
		
ret, thresh = cv2.threshold(thresh,127,255,0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
done = False
# Execute remove blobs function
output,image = init_remove_blobs(thresh)

# Display image with 3 channels which has the rectangles of the contours
cv2.imshow("OUTPUT",output)
# Display image with binary channel without any drawing
cv2.imshow("NEW_THRESH",image)


thresh = image

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
