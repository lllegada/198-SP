
#Find out why OTSU method doesn't work
from random import shuffle
import numpy as np 	
import math
from matplotlib import pyplot as plt
from sklearn import svm, datasets
import os
import skimage
from skimage import io
from PIL import Image
import cv2
import glob
from sklearn import model_selection
from sklearn.externals import joblib
from revised_func import remove_coinsides
import sys
from utils import resizeTo20,resizeTo28,transform20x20

# {'C': 100, 'gamma': 0.01}
# 0.927754677755
def feat_extract(folder_path):
	num = 0
	data=[]
	labels=[]

	for root, dirs, files in os.walk(folder_path):
		for image_path in files:   
			image = cv2.imread(folder_path + "/" + image_path,0)
			label = image_path[:1]     
			# cv2.imshow("ORIGINAL",image)
			# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			# image = cv2.threshold(image, 0, 255,
			# cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
			# cv2.imshow("BINARIZED",image)
			# cv2.waitKey(0)
			# image = resizeTo20(image)
			# print("image.shape: ",image.shape)
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
			# cv2.imshow("ORIGINAL: ",image)
			# print("IMAGE NAME:", image_path)
			# centered = transform20x20(image)
			# fin = resizeTo28(centered)
			fin = image
			# print("Extractin feature....")
			winSize = (28,28)
			blockSize = (8,8)
			blockStride = (4,4)
			cellSize = (4,4)
			nbins = 9
			derivAperture = 1
			winSigma = -1.
			histogramNormType = 0
			L2HysThreshold = 0.2
			gammaCorrection = 1
			nlevels = 28
			signedGradient = True
			 
			hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient) 
			# hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, useSignedGradients)
			descriptor = hog.compute(fin)  
			
			
			data.append(descriptor)
			##Labels contains all the labels of the image ..see line 46
			labels.append((label))


		# ensemble
	return data,labels
def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 100]
    gammas = [0.001, 0.01, 0.1, 10]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = model_selection.GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    grid_search.best_score_
    # print(grid_search.cv_results_)
    return grid_search.best_params_,grid_search.best_score_
if __name__ == '__main__':
	#Change the value inside feat extract into the name of your folder containing the training set (only works if inside Python directory)
	train_data,train_labels = feat_extract("final_prep")
	train_data = np.squeeze(train_data) #Equivalent of flatten except that whole array
	
	####convert data and labels into a numpy array 
	train_data = np.array(train_data)
	train_labels = np.array(train_labels)
	print(train_data.shape)
	print (train_labels.shape)


	#Use these train and test values for prediction,training,testing
	X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_data, train_labels, test_size=0.20, shuffle=True)

	# Try the best parameters
	# best_params,best_score = svc_param_selection(X_train,Y_train,10)
	
	# print(best_params)
	# print(best_score)
	# sys.exit(0)
	# TRAINING
	print("X_train: ",X_train.shape)
	clf_small = svm.SVC(gamma = 0.01, C = 100,probability=True)
	# print("Unique: ", np.unique(train_labels))


	#Uncomment this if you need to train your model with the best params value 
	clf_small.fit(X_train,Y_train)
	
	
	# X_test,Y_test = feat_extract("tesi")
	X_test = np.squeeze(X_test) #Equivalent of flatten except that whole array
	print("Xtest: ", len(X_test))
	print(X_test.shape)
	# print(X_test[0])
	# print("Ytest: ",Y_test[0])
	# import sys; sys.exit()
	#======  Uncomment this if you are going to test your model with the model  
	# prediction = clf2.predict(X_test[0].reshape(1,-1))
	# scoring = clf2.score(X_test[0].reshape(1,-1), Y_test[0].reshape(1,-1))

	joblib_file = "cursive_small_model.pkl"  
	joblib.dump(clf_small, joblib_file)
	# joblib.dump(clf2, joblib_file)
	# print(Y_test)
	# print(prediction) 
	# print(scoring)

	# uncomment if you are going to train only one sample\
	# X_test = np.squeeze(X_test)
	# Y_test = np.squeeze(Y_test)
	# X_test = X_test.reshape(1,-1)
	# Y_test = Y_test.reshape(1,-1)


	loaded_model = joblib.load("cursive_small_model.pkl")
	# print(loaded_model.__dict__.keys())
	# import sys; sys.exit()
	pred = loaded_model.predict(X_test)
	print("Ytest: ", Y_test)
	print(pred)
	result = loaded_model.score(X_test, Y_test)
	print(result)
	        # =====================================
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	exit(0)