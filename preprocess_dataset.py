
import numpy as np 	
import math
from matplotlib import pyplot as plt
import os
import cv2
import glob
import sys
from utils import resizeTo20,resizeTo28,transform20x20





def preprocess_dataset(folder_path,folder_dest):
	# folder_path = "ALL_new"
	# folder_dest = "post_otsu"
	num = 0

	for root, dirs, files in os.walk(folder_path):
		for image_path in files:   
			image = cv2.imread(folder_path + "/" + image_path)
			label = image_path[:1]     
			image = resizeTo20(image)
			print("image.shape: ",image.shape)
			# image = cv2.resize(image, (20,20), interpolation = cv2.INTER_AREA)
			# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			# bitwise = cv2.bitwise_not(gray)
			# cv2.imshow("GRAY: ",gray)   
			# cv2.imshow("BITWISE: ",bitwise)       

			# threshold the image, setting all foreground pixels to
			# 255 and all background pixels to 0
			# thresh = cv2.threshold(bitwise, 0, 255,
			# 	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
			# cv2.imshow("thresh",thresh)
			# rows,cols = thresh.shape
			# print("rows: {} columns: {}" .format(rows,cols) )
			cv2.imshow("ORIGINAL: ",image)
			print("IMAGE NAME:", image_path)
			centered = transform20x20(image)
			fin = resizeTo28(centered)
			print("finshape:" ,fin.shape)
			print(folder_dest + "/" + label + "_"+ str(num) +".jpg" )
			cv2.imwrite(folder_dest + "/" + label + "_"+ str(num) +".jpg" ,fin)
			num = num + 1                   
			# cv2.imshow("ORGIIANL: ",image)
			cv2.imshow("PINAL: ",fin)                                            
			cv2.waitKey(1)
			# sys.exit(0)
	# cv2.waitKey(0)
	cv2.destroyAllWindows()
	exit(0)


if __name__ == '__main__':
	print("I'm in the main and I am doing nothing. Better checkout the code.")
	preprocess_dataset("ALL_cursive_small","final_prep")
# Find out why your method doesn't work for all data even if same naman yung shape nila     