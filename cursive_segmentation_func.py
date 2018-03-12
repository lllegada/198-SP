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

# this function finds the difference between 2 adjacent columns
# Returns the columns to be removed from the PSC
# Far columns are to be represented as one column na lang
def find_differences(PSC):
	oversegmentedPSC = []
	difference = []
	for i, val in enumerate(PSC):
		if i > 0:
			dif = PSC[i] - PSC[i-1]
			difference.append(dif)
			if dif > 1:
				oversegmentedPSC.append(PSC[i-1])

	thresh = int(np.mean(difference))



	return oversegmentedPSC, thresh

# removes oversegmentedPSC from the PSC array
def remove_initial_oversegmentation(oversegmentedPSC, PSC):
	for x in oversegmentedPSC:
		PSC.remove(x)

	return PSC

# gets threshold for segmented column 
def get_thresh(PSC):
	difference = []
	for i, val in enumerate(PSC):
		if i > 0:
			dif = PSC[i] - PSC[i-1]
			difference.append(dif)
	# Experiment with the multiplier to achieve a more generalized result
	multiplier = 0.65
	thresh = int(np.median(difference) * multiplier)

	return thresh,difference

def skeletonization(image):
	done = False
	
	skel = np.zeros(image.shape,np.uint8)
	size = np.size(image)
	element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
	while(not done):	
		eroded = cv2.erode(image, element)
		temp = cv2.dilate(eroded, element)
		temp = cv2.subtract(image, temp)
		skel = cv2.bitwise_or(skel,temp)
		image = eroded.copy()

		zeros = size - cv2.countNonZero(image)

		if zeros == size:
			done = True

	return image,skel

# Removes PSCs that are greater than the threshold
# I actually can't quite understand this
def remove_nonPSC(PSC_values,thresh):
	SC = []
	for i, val in enumerate(PSC_values):
		if i > 0:
			dif = PSC_values[i] - PSC_values[i-1]
			if dif > thresh:
				SC.append(PSC_values[i-1])
			if i == len(PSC_values)-1:
				SC.append(PSC_values[i-1])
	return SC

def display_colored_image(skel,psc_image,final,last):	
	skel = cv2.cvtColor(skel, cv2.COLOR_GRAY2RGB)
	psc_image = cv2.cvtColor(psc_image, cv2.COLOR_GRAY2RGB)
	final = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)
	last = cv2.cvtColor(last, cv2.COLOR_GRAY2RGB)

	return skel,psc_image,final,last

def get_image_shape(skel):
	height, width = skel.shape
	return height,width

def draw_lines(height,columns,image,b,g,r):
	for x in columns:
		point1 = (x, 0)
		point2 = (x, height)
		cv2.line(image, point1, point2, (b, g, r), 1)

img = cv2.imread("1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
thresh = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]	
ret, thresh = cv2.threshold(thresh,127,255,0)
done = False
# Execute remove blobs function
output,image = init_remove_blobs(thresh)

# Display image with 3 channels which has the rectangles of the contours
cv2.imshow("OUTPUT",output)
# Display image with binary channel without any drawing
cv2.imshow("NEW_THRESH",image)


image,skel = skeletonization(image)




# ===================================================
# Duplicates of the original's black and white image

psc_image = skel
final = skel
last = skel
skel_img = skel

height, width = get_image_shape(skel_img)


# find the number of 1s in each column
nonZeroYCount = np.count_nonzero(skel, axis = 0)



skel,psc_image,final,last = display_colored_image(skel,psc_image,final,last)
# gets the index of columns which contains 0 or 1 values that might be PSCs
PSC_values = [i for i,val in enumerate(nonZeroYCount) if (val <= 1)] 

# draw line on PSC through the index from PSC_values	
draw_lines(height,PSC_values,skel,0,0,255)

# 
removePSC, differences= find_differences(PSC_values)
PSC_values = remove_initial_oversegmentation(removePSC, PSC_values)


# shows image with segmented columns
draw_lines(height,PSC_values,psc_image,255,0,0)

# merges PSC
thresh,differences = get_thresh(PSC_values)

# Remove values
SC = remove_nonPSC(PSC_values,thresh)

# magic number threshold
thresh = int(max(differences)/5)
SSC = remove_nonPSC(SC,thresh)
draw_lines(height,SSC,final,0,255,0)


thresh, differences = get_thresh(SSC)
FinalSSC = remove_nonPSC(SSC,thresh)
draw_lines(height,FinalSSC,last,220,255,0)

print("FinalSSC values: ", FinalSSC)
cv2.imshow("skeleton", skel)
cv2.imshow("SC", psc_image)
cv2.imshow("SSC", final)
cv2.imshow("FINALSSC", last)
cv2.waitKey(0)
cv2.destroyAllWindows()
