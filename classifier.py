import os
import glob
import cv2
import caffe
import pickle
import numpy as np 
from caffe.proto import caffe_pb2
from string import ascii_lowercase

caffe.set_mode_gpu()

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
BINARY_PROTO_PATH = '/home/lincy/caffe/data/SP/sp_mean.binaryproto'
DEPLOY_PROTOTXT = '/home/lincy/caffe/models/sp_lenet/lenet_deploy.prototxt'
CAFFE_MODEL = '/home/lincy/caffe/models/sp_lenet/caffe_lenet_train_iter_10000.caffemodel'
VALIDATION_FILE_PATH = "/home/lincy/caffe/data/SP/dummy.txt"
IMG_PATH = "/home/lincy/caffe/data/SP/val/"
CLASSIFICATION_FILE_PATH = "logs/Predictions/Lenet/"

# def net_to_python(prototxt, caffemodel):
# 	net = caffe.Net(prototxt, caffemodel, caffe.TEST) # read the net + weights
# 	pynet_ = [] 
# 	for li in xrange(len(net.layers)):  # for each layer in the net
# 		layer = {}  # store layer's information
# 		layer['name'] = net._layer_names[li]
# 		# for each input to the layer (aka "bottom") store its name and shape
# 		layer['bottoms'] = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape) 
# 		             for bi in list(net._bottom_ids(li))] 
# 		# for each output of the layer (aka "top") store its name and shape
# 		layer['tops'] = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape) 
# 		          for bi in list(net._top_ids(li))]
# 		layer['type'] = net.layers[li].type  # type of the layer
# 		# the internal parameters of the layer. not all layers has weights.
# 		layer['weights'] = [net.layers[li].blobs[bi].data[...] 
# 		            for bi in xrange(len(net.layers[li].blobs))]
# 		pynet_.append(layer)
# 	return pynet_


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
	net = caffe.Net(deploy, model, caffe.TEST)

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

		# path = IMG_PATH + filename[0]
		path = "data/SP/val/x_3908.jpg"
		img = cv2.imread(path, cv2.IMREAD_COLOR)
		img = transform_img(img, IMAGE_WIDTH, IMAGE_HEIGHT)

		net.blobs['data'].data[...] = transformer.preprocess('data', img)
		out = net.forward()
		pred_probas = out['prob']

		test_ids = test_ids + [path.split('/')[-1][:-4]]
		preds = preds + [pred_probas.argmax()]
		correctLabels = correctLabels + [filename[1].rstrip("\r\n")]

		counter = counter + 1

		for i, val in enumerate(preds):
			prediction = "prediction: " + str(val)
			label = "correct label: " +  str(correctLabels[i])

		if pred_probas.argmax() == int(filename[1].rstrip("\r\n")):
			correct = correct + 1

		letter = generate_letter(val)
		print "Iteration ", counter , "\n"
		print prediction, "\t letter: " + letter
		print label

		# write results to a txt file
		store_output(prediction, label, counter, letter)

	return correct, counter

# a dictionary is generated forming a key:value pair (ex. {0:'a', 1:'b', 2:'c', ...})
def generate_letter(val):
	lettermap = dict((number, char) for number, char in enumerate(ascii_lowercase, 0))
	letter = lettermap[val]
	return letter

# writes the prediction and correct answer in a text file name 'classification' 
def store_output(prediction, label, counter, letter):
	file = open(CLASSIFICATION_FILE_PATH+"one_img.txt", "a+")
	file.write(prediction + "\n" + label + "\t" + "letter: "+ letter + "\n" + "Iteration: " + str(counter) + "\n\n")
	file.close()

# accuracy = number of correct predictions/total number of images in validation set
def check_accuracy(correct, counter):
	accuracy = float(correct)/float(counter)
	print "Accuracy: ",accuracy

	file = open(CLASSIFICATION_FILE_PATH+"one_img.txt", "a+")
	file.write("\n" + "Accuracy: " + str(accuracy))
	file.close()

transformer, net = set_up(BINARY_PROTO_PATH, DEPLOY_PROTOTXT, CAFFE_MODEL)
correct, counter = classifier(VALIDATION_FILE_PATH, transformer, net)
check_accuracy(correct, counter)
# pynet = net_to_python(DEPLOY_PROTOTXT, CAFFE_MODEL)
# print "\n\n PYNET: ", pynet


