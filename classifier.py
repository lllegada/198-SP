import os
import glob
import cv2
import caffe
import pickle
import numpy as np 
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from string import ascii_lowercase

caffe.set_mode_gpu()

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
CAFFE_ROOT = '/home/lincy/caffe/'
BINARY_PROTO_PATH = '/home/lincy/caffe/data/final_sp_temp/sp_mean.binaryproto'
DEPLOY_PROTOTXT = '/home/lincy/caffe/models/sp_final_lenet_temp/lenet_deploy.prototxt'
CAFFE_MODEL = '/home/lincy/caffe/models/sp_final_lenet_temp/exp_10/caffe_lenet_fixed10_train_iter_10000.caffemodel'
VALIDATION_FILE_PATH = "/home/lincy/caffe/data/final_sp_temp/val.txt"
IMG_PATH = "/home/lincy/caffe/data/final_sp_temp/val/"
CLASSIFICATION_FILE_PATH = "logs/final_sp_temp/exp_10/"


# resizing the image according the the set width and height
def transform_img(img, width, height):
	img = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)
	return img


# read mean image, set caffe model and weights
def set_up(binary, deploy, model):
	# Mean Image
	mean_blob = caffe_pb2.BlobProto()
	with open(binary) as f:
		mean_blob.ParseFromString(f.read())
	mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape((mean_blob.channels, mean_blob.height, mean_blob.width))

	# Model architecture and trained model's weights
	# net = caffe.Net(deploy, model, caffe.TEST)
	net = caffe.Classifier(deploy, model)

	# Image transformers
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_mean('data', mean_array)
	transformer.set_transpose('data',(2,0,1))

	return transformer, net

# predicts the label of the characters
def classifier(filepath, transformer, net):
	with open(filepath) as f:
		contents = f.readlines()

	# Making predictions
	test_ids = []
	preds = []
	correctLabels = []
	counter = 0
	correct = 0

	for con in contents:
		filename = con.split(" ")

		path = IMG_PATH + filename[0]

	
		img = cv2.imread(path, cv2.IMREAD_COLOR)
		img = transform_img(img, IMAGE_WIDTH, IMAGE_HEIGHT)


# ==============================================================
	
		input_image = caffe.io.load_image(path)
		prediction = net.predict([input_image])
	


		print "========================================================="

		# returns the probability and the top 3 predictions

		proba = prediction[0][prediction[0].argmax()]
		ind = prediction[0].argsort()[-3:][::-1]


		net.blobs['data'].data[...] = transformer.preprocess('data', img)
		out = net.forward()
		pred_probas = out['prob']


		test_ids = test_ids + [path.split('/')[-1][:-4]]
		preds = preds + [pred_probas.argmax()]
		correctLabels = correctLabels + [filename[1].rstrip("\r\n")]
		

		counter = counter + 1

		for i, val in enumerate(preds):
			pred= "prediction: " + str(val)
			label = "correct label: " +  str(correctLabels[i])

		if pred_probas.argmax() == int(filename[1].rstrip("\r\n")):
			correct = correct + 1

		letter = generate_letter(val)
		print "Iteration ", counter , "\n"

		print "top 3: ", ind
		print "probability: ", proba
		print pred, "\t letter: " + letter
		print label



		# UNCOMMENT to write results to a txt file
		# store_output(pred, label, counter, letter)

	return correct, counter

# a dictionary is generated forming a key:value pair (ex. {0:'a', 1:'b', 2:'c', ...})
def generate_letter(val):
	# lettermap = dict((number, char) for number, char in enumerate(ascii_lowercase, 0))
	lettermap = { 0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'j', 10:'k', 11:'l', 12:'m', 13:'n', 14:'o', 15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u', 21:'v', 22:'w', 23:'x', 24:'y', 25:'z', 26:'A', 27:'B', 28:'C', 29:'D', 30:'E', 31:'F', 32:'G', 33:'H', 34:'I', 35:'J', 36:'K', 37:'L', 38:'M', 39:'N', 40:'O', 41:'P', 42:'Q', 43:'R', 44:'S', 45:'T', 46:'U', 47:'V', 48:'W', 49:'X', 50:'Y', 51:'Z' }

	letter = lettermap[val]
	return letter

# writes the prediction and correct answer in a text file name 'classification' 
def store_output(prediction, label, counter, letter):
	file = open(CLASSIFICATION_FILE_PATH+"prediction10.probs.txt", "a+")
	file.write(prediction + "\n" + label + "\t" + "letter: "+ letter + "\n" + "Iteration: " + str(counter) + "\n\n")
	file.close()

# accuracy = number of correct predictions/total number of images in validation set
def check_accuracy(correct, counter):
	accuracy = float(correct)/float(counter)
	print "Accuracy: ",accuracy

	file = open(CLASSIFICATION_FILE_PATH+"scoreprobs.txt", "a+")
	file.write("\n" + "Accuracy: " + str(accuracy))
	file.close()

transformer, net = set_up(BINARY_PROTO_PATH, DEPLOY_PROTOTXT, CAFFE_MODEL)
correct, counter = classifier(VALIDATION_FILE_PATH, transformer, net)
check_accuracy(correct, counter)


