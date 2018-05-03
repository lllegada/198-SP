# outputs a text file containing filenames of images and their corresponding label.
# The text file will then be used for parts.sh

import os
import glob
import math
from random import shuffle
def store_output(out, kind):
	file = open(kind + ".txt","a+")
	out = out.split("/")[1]
	print out
	file.write(out + "\n")
	file.close()
	
def getname_tool(folder_path, n, partition, kind):
	count = 0
	n = 0
	
	for root,dirs,files in os.walk(folder_path):
		# Uncomment this if you are working with ordered categories such as alphabets,digits, etc.
		dirs.sort()
		train = math.ceil(len(files)*partition)
		
		if (len(files)!=0):
			shuffle(files)
			for f in files:				
				print f
				if count < train:
					parent = root.split("/")[1]
					output = (parent + "/" + f + " "+ str(n))
					store_output(output, kind)
					count = count + 1
			n = n+1
			count = 0

if __name__ == '__main__':
	# the images must be sorted into folders (ex. a, b , c)
	# # TRAIN
	getname_tool('dataset_orig/dataset_lowercase/', 0, 0.7, 'train')
	getname_tool('dataset_orig/dataset_uppercase/', 26, 0.7, 'train')

	# # VAL
	getname_tool('dataset_orig/dataset_lowercase/', 0, 0.1, 'val')
	getname_tool('dataset_orig/dataset_uppercase/', 26, 0.1, 'val')

	