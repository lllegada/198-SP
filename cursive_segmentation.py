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

# this function finds the difference between 2 adjacent pixels
def find_differences(PSC, oversegmentedPSC, difference, threshold):

	for i, val in enumerate(PSC):
		if i > 0:
			dif = PSC[i] - PSC[i-1]
			difference.append(dif)
			if dif > 1:
				oversegmentedPSC.append(PSC[i-1])

	thresh = int(np.mean(difference))

	print("PSC_values: ", PSC)


	return PSC, oversegmentedPSC, difference, thresh

# removes oversegmentedPSC from the PSC array
def remove_redlines(oversegmentedPSC, PSC):
	for x in oversegmentedPSC:
		PSC.remove(x)

	return oversegmentedPSC, PSC

# gets threshold for segmented column 
def get_thresh(PSC, difference):
	for i, val in enumerate(PSC):
		if i > 0:
			dif = PSC[i] - PSC[i-1]
			difference.append(dif)

	thresh = int(np.mean(difference))

	print("PSC_values: ", PSC)


	return thresh


img = cv2.imread("rules.png")
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
psc_image = skel
final = skel

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
psc_image = cv2.cvtColor(psc_image, cv2.COLOR_GRAY2RGB)
final = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)

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

removePSC = []
differences = []
PSC_values, removePSC, differences, thresh = find_differences(PSC_values, removePSC, differences, thresh)

print("THRESHOLD: ", thresh)

removePSC, PSC_values = remove_redlines(removePSC, PSC_values)

print("differences: ", differences)
print("NEW PSC_values: ", PSC_values)


# shows image with segmented columns
for x in PSC_values:
	point1 = (x, 0)
	point2 = (x, height)
	cv2.line(psc_image, point1, point2, (255, 0, 0), 1)

# merges PSC
thresh = get_thresh(PSC_values, differences)

print("NEW THRESH: ", thresh)

SC = []
SSC = []

for i, val in enumerate(PSC_values):
		if i > 0:
			dif = PSC_values[i] - PSC_values[i-1]
			if dif > thresh:
				SC.append(PSC_values[i-1])

thresh = int(max(differences)/2)
# thresh = int(np.mean(differences))

for x in SC:
	point1 = (x, 0)
	point2 = (x, height)
	cv2.line(final, point1, point2, (100, 0, 255), 1)

for i, val in enumerate(SC):
		if i > 0:
			dif = SC[i] - SC[i-1]
			if dif > thresh:
				SSC.append(SC[i-1])

for x in SSC:
	point1 = (x, 0)
	point2 = (x, height)
	cv2.line(final, point1, point2, (0, 255, 0), 1)



cv2.imshow("skeleton", skel)
cv2.imshow("SC", psc_image)
cv2.imshow("FINAL", final)
cv2.waitKey(0)
cv2.destroyAllWindows()
