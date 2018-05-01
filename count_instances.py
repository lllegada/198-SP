
import glob
import cv2
import numpy as np 	
import math

import os




def count_instances(folder_path):
	numfiles = 0
	num = 0
	for root, dirs, files in os.walk(folder_path):
		# Get the first character of each file's name
		instaces_count = [file[0] for file in files]
		# print("Contents: ",instaces_count)
		alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
		for letter in alphabet:
			print(letter + ": ",instaces_count.count(letter))






			
			

# Replace folder name (must be in the same directory as the code)
count_inst = count_instances("ALL_DATASET")