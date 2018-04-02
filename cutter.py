# Code must ONLY be used if image has been manually cropped to contain only the bond paper part
# DO NOT USE for freshly captured images that have not been cropped yet.




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
from utils import get_median_area

def cut_dataset(folder_path):
	numfiles = 0
	
	for root, dirs, files in os.walk(folder_path):
		for image_path in files:   
			n=0
			numfiles = numfiles + 1
			image_name = image_path[0]
			print("\nimage_path: ",image_path)
			image = cv2.imread(folder_path + "/" + image_path)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


			# Smooth image
			blur = cv2.GaussianBlur(gray, (3, 3), 0)
			filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 77, 3)

			# Some morphology to clean up image,originally 7,7
			kernel = np.ones((3,3), np.uint8)
			opening = cv2.morphologyEx(filtered,cv2.MORPH_OPEN, kernel, iterations = 2)
			closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations =2)

			cv2.namedWindow('filtered',cv2.WINDOW_NORMAL)
			cv2.namedWindow('opening',cv2.WINDOW_NORMAL)
			cv2.imshow("filtered",filtered)
			cv2.imshow("closing",closing)
			cv2.imshow("opening",opening)


			_, contours0, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			sliding = []
			contours2 =[]
			localCut = []
			differences=[]
			globalCut = [] #stores indices of differences that has 50% chance of being a space
			words = []
			contours=[]
			current_area = 0

			temp = 0

			indices=[] #stores values of the contours that are co-inside the contour
			for c in contours0:
				[x, y, w, h] = cv2.boundingRect(c)
				current_area = w*h
				# print("current area: ",current_area)
				temp = temp + current_area
			ave_area = temp/len(contours0)

			threshold_area = (0.05*(ave_area))

			for c in contours0:
				[x,y,w,h] = cv2.boundingRect(c)
				if ((w*h) >= threshold_area ):
					contours.append(c)

			for ic,c in enumerate(contours):
				x, y, w, h = cv2.boundingRect(c)
				
			for c in contours:
				x, y, w, h = cv2.boundingRect(c)

			indices=[] #stores values of the contours that are co-inside the contour
			for c in contours:
				[x, y, w, h] = cv2.boundingRect(c)
				for index,cn in enumerate(contours):
					[i,j,k,l] = cv2.boundingRect(cn)
					if ((i < (x+w) and (i > x)) and ((i+k) < (x+w) and (i+k) > x )):
						if((((j+l) < (y+h)) and ((j+l) > y )) and ((j < (y+h)) and (j > y))):
							indices.append(index)

			contours2 = [c for i,c in enumerate(contours) if i not in indices]
			print("Length of contours2: ",len(contours2))
			for c in contours2:
				x, y, w, h = cv2.boundingRect(c)
				height = y+h
				width = x+w
				roi = image[y:height, x:width]
				
				path = "C:/Users/User/AppData/Local/Programs/Python/Python36-32/shebna_big_initial/"
				cv2.imwrite(path + image_name + "_m_"+str(n)  + ".jpg", roi)
				n = n+1
			cv2.namedWindow('final',cv2.WINDOW_NORMAL)
			cv2.imshow('final',image)
			# cv2.waitKey(0)
		print("Number of files: ", numfiles)



# Replace folder name (must be in the same directory as the code)
cut_files = cut_dataset("tobecut")