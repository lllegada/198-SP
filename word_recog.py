from sklearn import model_selection
from sklearn.externals import joblib
from random import shuffle
import numpy as np 	
import math
import sys
import cv2
from utils import remove_coinsides,remove_noise,sort_LR

# Use for getting the name of the file
# Name of the file will be used as a true label
class ImageName:
    def __init__(self, img_name):
        self.img = cv2.imread(img_name)
        self.__name = img_name

    def __str__(self):
        return self.__name

# Preprocessing technique using OTSU
def otsu_preprocess(image):	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	prepimage = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	_, contours, hierarchy = cv2.findContours(prepimage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	return prepimage,contours

# Displays the contours on the input image
def show_contours(contours,image):
	for c in contours:
		x,y,w,h = cv2.boundingRect(c)
		print(x,y,w,h)
		cv2.rectangle(image, (x, y), (x+w, y+h), (145, 100, 0), 2)
		cv2.imshow("IMAGE",image)
		cv2.waitKey()

def feat_extract(image,Y_test):
	data = []
	labels = []
	label = Y_test

	_, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	contours2 = remove_coinsides(contours)
	

#####------START OF MACHINE LEARNING AND FEATURE EXTRACTION-------####

	print("Number of ROI: ",len(contours2))


	#Uncomment for error checking if more than 1 roi was detected for recognition
	# Feature extractor must only detect 1 roi
	# if(len(contours2) > 1):
	# 	for c in contours2:
	# 		[x, y, w, h] = cv2.boundingRect(c)
	# 		cv2.rectangle(image, (x, y), (x+w, y+h), (145, 100, 0), 4)
	# 		cv2.imshow("ROI",image)
	# 		sys.exit()



	# Pad image just in case it is not 64x64
	for c in contours2:
		[x, y, w, h] = cv2.boundingRect(c)
		if(h > w):
			crop1 = image[y:y+h, x:(x+w)]
			diff = (h)-(w)
			padleft = math.ceil(diff/2)
			padright = math.floor(diff/2)
			padded_img = np.zeros((h,h),np.uint8)
			
			padded_img[:,padleft:h-padright] = crop1
		elif(h < w):
			crop1 = image[y:y+h, x:(x+w)]
			diff = w-h
			padtop = math.ceil(diff/2)
			padbottom = math.floor(diff/2)
			padded_img = np.zeros((w,w),np.uint8)
			padded_img[padtop:w-padbottom,:] = crop1
		else:
			padded_img = image
		crop1 = cv2.resize(padded_img,(64,64))
		
		# Set parameters for HOG
		winSize = (64,64)
		blockSize = (16,16)
		blockStride = (8,8)
		cellSize = (8,8)
		nbins = 9
		derivAperture = 1
		winSigma = -1.
		histogramNormType = 0
		L2HysThreshold = 0.2
		gammaCorrection = 1
		nlevels = 64
		signedGradient = True
		 
		hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient) 
		# hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, useSignedGradients)
		descriptor = hog.compute(crop1)  
		
		
		data.append(descriptor)
		##Labels contains all the labels of the image
		labels.append((label))
		print("LABEL: ",label)
	return data,labels


# Calculates the score for all the letters.
def calc_total_score(true_labels,Ppred):
	correct = 0
	print("ALL PREDS: ",Ppred)
	for i,label in enumerate(true_labels):
		if label == Ppred[i]:
			correct = correct + 1
	fscore = correct/len(Ppred)
	print("FINAL SCORE: ", fscore)

def disp_word(Ppred):
	finalString = ''.join(Ppred)
	print(finalString)

if __name__ == '__main__':
	Ppred = []
	x = ImageName("LLLLLKKK.jpg") # Change the file name
	image_filename = str(x)
	image = cv2.imread(image_filename)
	image_string = image_filename.split('.')[0]
	true_labels = list(image_string)
	
	# Resize image so it'll fit on screen
	h,w = image.shape[:2]
	ar = w / h
	nw = 600 
	nh = int(nw / ar)
	image = cv2.resize(image,(nw,nh))
	

	#--------Preprocessing starts here ------#
	prepimage,contours = otsu_preprocess(image)
	contours1 = remove_noise(contours)
	contours2 = remove_coinsides(contours1)
	show_contours(contours2,image)


	# Get the corresponding x,y,w,h points of each contour
	# This is necessary for sorting the letters L-R
	contours3 = [cv2.boundingRect(c) for c in contours2]

	# Contours are sorted L-R for proper prediction results
	contours3 = sort_LR(contours3)


	i=0
	for c in contours3:
		if(i < len(contours2)):
		
			x,y,w,h = c[0],c[1],c[2],c[3]
			print(x,y,w,h)
			cv2.rectangle(image, (x, y), (x+w, y+h), (145, 100, 0), 2)
			letter = prepimage[y:y+h,x:x+w]
			
			
			X_test,Y_test = feat_extract(letter,true_labels[i])
			
			# Transform to numpy array
			X_test = np.squeeze(X_test)
			Y_test = np.squeeze(Y_test)
			
			# Reshape to 1,-1 because there is only a single sample
			X_test = X_test.reshape(1,-1)
			Y_test = Y_test.reshape(1,-1)
			
			# Load the model we previously trained
			# This uses the joblib library from sklearn
			loaded_model = joblib.load("clf2_model.pkl")
			pred = loaded_model.predict(X_test)
			result = loaded_model.score(X_test, Y_test)
			
			# Compile every single recognition to a list
			Ppred.append(pred.tolist()[0])
			
			cv2.imshow("letter",letter)
			cv2.waitKey(0)
		i = i + 1
	
	calc_total_score(true_labels,Ppred)
	disp_word(Ppred)
		
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	exit(0) 