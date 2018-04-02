"""
	W A R N I N G :
	*** DO NOT CHANGE ANYTHING IN THIS FILE
		UNLESS YOU ARE ADDING A NEW FUNCTION
		for local experimentation:
			- duplicate the function in your python file 

	All global functions should be placed here
	in order to reduce redundancy of its presence
	for every file.
"""




import cv2
import glob
import numpy as np
import math
from sklearn import model_selection,svm
import sys
import statistics

def resizeTo28(img):
	r = 28.0 / img.shape[1]
	dim = (28, int(img.shape[0] * r))
	res = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	cv2.imshow("To28",res)
	cv2.waitKey(1)
	return res
def resizeTo20(img):
	# r = 20.0 / img.shape[1]
	# dim = (20, int(img.shape[0] * r))
	res = cv2.resize(img, (20,20), interpolation = cv2.INTER_AREA)
	cv2.imshow("To20",res)
	cv2.waitKey(1)
	return res
def get_ROI(image,contours):
	# for c in contours:
	x,y,w,h = cv2.boundingRect(contours[0])
	# print(x,y,w,h)
	patch = image[y:(y+h),x:(x+w)]
	

	return patch
		

def center_image(patch,contours):

	new_img = np.zeros((20,20),np.uint8)
	# print(new_img.shape)
	# print(patch.shape)
	# x,y,w,h = cv2.boundingRect(contours[0])
	h,w = patch.shape[0:2]
	mX = math.floor((20-w)/2)
	mY = math.floor((20-h)/2)
	# print("mX: {}  mY:{}  " .format(mX,mY))
	x1 = mX
	x2 = x1+w
	y1 = mY
	y2 = y1+h
	# print("y1: {}  y2: {}  x1: {}  x2: {}"  .format(mY,y2,x1,x2))
	new_img[y1:y2,x1:x2] = patch

	# print("--------------")
	# cv2.imshow("CENTER",new_img)
	# cv2.waitKey(0)
	return new_img

def transform20x20(image):
	# _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	original = image
	prepimage,contours = otsu_preprocess(image)
	contours1 = remove_noise(contours)
	contours = remove_coinsides_letter(contours1)



	# if(len(contours) == 1):
		# Center the position of the image to a new 28x28 image
		
		

	if(len(contours) >1 ):
		# contours = sort_LR(contours)
		contours = remove_coinsides_letter(contours)
		# show_contours(contours,image)
		print("More than 1 contour found. Hopefully this was fixed.")		
		# cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 40), 1)
		# cv2.imshow("SHOWCONTOURS",image)
		# cv2.waitKey(0)
		# # cv2.waitKey(0)
		# debug = cv2.resize(image, (500,500), interpolation = cv2.INTER_AREA)
		# cv2.imshow("DEBUG",debug)
		# cv2.waitKey(0)
		# sys.exit(0)

	patch = get_ROI(prepimage,contours)
	centered = center_image(patch,contours)
	cv2.imshow("patch",patch)
	cv2.imshow("centered",centered)
	cv2.waitKey(1)
	
	return centered
def get_max_area_contour(contours):
	areas = []
	for i,c in enumerate(contours):
		[x, y, w, h] = cv2.boundingRect(c)
		current_area = w*h
		areas.append(current_area)
	contour_roi_index = areas.index(max(areas))
	contour_roi = contours[contour_roi_index]
	# print("average area: ",ave_area)
	return contour_roi
def get_ave_area(contours0):
	temp = 0
	for c in contours0:
		[x, y, w, h] = cv2.boundingRect(c)
		current_area = w*h
		# print("current area: ",current_area)
		temp = temp + current_area
	ave_area = temp/len(contours0)
	print("average area: ",ave_area)
	return ave_area
# def get_midpoint(contours):
# 	contours_length = len(contours)
# 	if(contours_length%2 == 0): #Even
# 		midpoint_index = contours_length/2
# 	else: #Odd
# 		mid_initial_index = contours_length/2
# # 		midpoint =  (contours[mid_initial_index] + contours[mid_initial_index+1])/2
	
# 	return contours[midpoint_index]
def get_median_area(contours0):
	# Sort the area ascending
	# Get the length of the sorted contours
	# find the midpoint
	# get the midpoint's area
	areas = []
	contours = [cv2.boundingRect(c) for c in contours0]
	# for c in contours0:
	# 	[x,y,w,h] = cv2.boundingRect(c)
	# 	sorted_contours = [x,y,w,h]
	sorted_contours = sort_LR(contours)
	for c in sorted_contours:
		[x,y,w,h] =c
		area = w*h
		areas.append(area)
	midpoint = statistics.median(areas)
	print(midpoint)
	return midpoint

def show_contours(contours,image):
	
	for c in contours:
		x,y,w,h = cv2.boundingRect(c)
		print(x,y,w,h)
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 40), 1)
		cv2.imshow("SHOWCONTOURS",image)
		cv2.waitKey(1)
		

def remove_noise(contours0):
	contours = []
	ave_area = get_ave_area(contours0)
	# print("average area: ",ave_area)
	threshold_area = (0.10*(ave_area))
	# print("threshold area: ",threshold_area)
	# contours = [c for cv2.boundingRect(c) in contours0 if (((c[0]+c[2])*(c[1]+c[3])) >= ave_area)]
	for c in contours0:
		[x,y,w,h] = cv2.boundingRect(c)
		if ((w*h) >= threshold_area ):
			contours.append(c)
	# print("Length of unfiltered contours: ",len(contours0))
	# print("Length of filtered contours: ",len(contours))
	return contours

def otsu_preprocess(image):
	gray = image
	print(len(image.shape))
	if(len(image.shape) > 2):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	prepimage = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	cv2.imshow("IN OTSU PREPREOCESS IMAGE",prepimage)
	cv2.waitKey(1)
	_, contours, hierarchy = cv2.findContours(prepimage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	return prepimage,contours

def remove_coinsides(contours):
	indices = []
	for c in contours:
		[x, y, w, h] = cv2.boundingRect(c)
		for index,cn in enumerate(contours):
			
			[i,j,k,l] = cv2.boundingRect(cn)
			
			if ((i < (x+w) and (i > x)) and ((i+k) < (x+w) and (i+k) > x )):
				# print("inside")
				if((((j+l) < (y+h)) and ((j+l) > y )) and ((j < (y+h)) and (j > y))):
			
					indices.append(index)
					

	contours2 = [c for i,c in enumerate(contours) if i not in indices]
	return contours2

def overlapX(boxA,boxB):
	Ax1,Ay1,Ax2,Ay2 = boxA
	Bx1,By1,Bx2,By2 = boxB

	end = Ax2
	start = Bx1
	if Bx1 < Ax1: # if B is before A
		end = Bx2
		start = Ax1

	return start <= end

def overlapY(boxA,boxB):
	Ax1,Ay1,Ax2,Ay2 = boxA
	Bx1,By1,Bx2,By2 = boxB

	end = Ay2
	start = By1
	if By1 < Ay1: # if B is before A
		end = By2
		start = Ay1

	return start <= end

def remove_coinsides_letter(contours):
	import itertools
	remove = []
	indices = range(len(contours))
	for index1,index2 in itertools.combinations(indices,2):
		contour1 = contours[index1]
		contour2 = contours[index2]

		c1_x1,c1_y1,w1,h1 = cv2.boundingRect(contour1)
		c2_x1,c2_y1,w2,h2 = cv2.boundingRect(contour2)
		c1_x2,c1_y2 = c1_x1 + w1, c1_y1 + h1 
		c2_x2,c2_y2 = c2_x1 + w2, c2_y1 + h2

		boxA = (c1_x1,c1_y1,c1_x2,c1_y2)
		boxB = (c2_x1,c2_y1,c2_x2,c2_y2)
		if overlapX(boxA,boxB) or overlapY(boxA,boxB):
			areaA = w1 * h1 
			areaB = w2 * h2 
			smaller = index1 # first box is smaller (assume)
			if areaB < areaA: # second box is smaller
				smaller = index2
			remove.append(smaller)

	return [contour for i,contour in enumerate(contours) if i not in remove]


def preprocess(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (3, 3), 0)
	filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 77, 3)

	# Some morphology to clean up image
	kernel = np.ones((3,3), np.uint8)
	opening = cv2.morphologyEx(filtered,cv2.MORPH_OPEN, kernel, iterations = 1)
	kernel2 = np.ones((3,3),np.uint8)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2, iterations =1)
	cv2.namedWindow('closing',cv2.WINDOW_NORMAL)
	cv2.namedWindow('res',cv2.WINDOW_NORMAL)
	cv2.namedWindow('filtered',cv2.WINDOW_NORMAL)
	cv2.namedWindow('opening',cv2.WINDOW_NORMAL)
	cv2.imshow("filtered",filtered)
	cv2.imshow("closing",closing)
	cv2.imshow("opening",opening)

	_, contours0, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	return contours0,closing


def sort_LR(contours):
	contours.sort(key=lambda b: b[0])
	return contours


def resize_img(thresh):
	h,w = thresh.shape[:2]
	ar = w / h 
	nw = 1300
	nh = int(nw / ar)        
	nimage = cv2.resize(thresh,(nw,nh))
	
	return nimage