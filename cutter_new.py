# Code must ONLY be used if image has been checked of perfect segmentation
# Check out the ## comments as a guide for checking perfect segmentation
# Instructions:
# --Input folder path of training images that are freshly captured
# -- Check out the ## comments
# -- Perfect segmentation means that there are no unwanted blobs captured by the cv2.rectangle



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
from utils import remove_noise,get_ave_area,resize_img,get_median_area

def cutter_new(folder_path):
	numfiles = 0
	
	for root, dirs, files in os.walk(folder_path):
			
		for image_path in files:   
			n=0
			numfiles = numfiles + 1
			image_name = image_path[0]
			print("\nimage_path: ",image_path)
			image = cv2.imread(folder_path + "/" + image_path)
			
			label = image_path.split("_")[0]
			print(label)


			gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
			_,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY) 
			nimage = resize_img(thresh)
			thresh = cv2.bitwise_not(thresh)
			nimage = resize_img(thresh)
			_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours

			contours = remove_noise(contours)
			average_area = get_ave_area(contours)
			print(len(contours))
			print (average_area)
			for contour in contours:
				[x,y,w,h] = cv2.boundingRect(contour)

				if((w*h) <= average_area):
					## Uncomment this if you're checking
					#cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
					height = y+h
					width = x+w
					roi = image[y:height, x:width]
					
					path = "C:/Users/User/AppData/Local/Programs/Python/Python36-32/shebna_big_initial/"
					## Comment this if you are checking
					cv2.imwrite(path + image_name + "_a_"+ str(n)  + ".jpg", roi)
					n = n+1
			image = resize_img(image)
			cv2.imshow("imahe",image)

			cv2.waitKey(1)


# Replace folder name (must be in the same directory as the code)
cut_files = cutter_new("All2") 