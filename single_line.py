from sklearn import model_selection
from sklearn.externals import joblib
from random import shuffle
import numpy as np 	
import math
import sys
import cv2
from utils import remove_coinsides,remove_coinsides_letter,remove_noise,sort_LR,get_max_area_contour,resizeTo20,resizeTo28,transform20x20, get_ave_area
import re

import glob 
from spellchecker import correction
import string
import heapq
import itertools as it
from ete3 import Tree, TreeStyle, TextFace, add_face_to_node
import pickle


# Use for getting the name of the file
# Name of the file will be used as a true label
class ImageName:
    def __init__(self, img_name):
        self.img = cv2.imread(img_name)
        self.__name = img_name

    def __str__(self):
        return self.__name

def closer_to_prev(contours3,index_c,x,y,w,h):
	new_x = contours3[index_c-1][0]
	new_y = contours3[index_c-1][1]
	new_w = contours3[index_c-1][2]
	new_h = contours3[index_c-1][3]
	# if(x < new_x):
	if(new_x < x):
		print("j case")
		new_w = new_w
		new_x = new_x
		new_h =  new_h +(new_y - y)
		new_y = y
		remove_index = index_c-1
	else:
		new_w = new_w + ((new_x+new_w)-(x+w))
		new_x = x
		new_h = new_h +(new_y - y)
		new_y = y
		remove_index = index_c-1

	return new_x,new_y,new_w,new_h,remove_index

def closer_to_next(contours3,index_c,x,y,w,h):
	new_x = contours3[index_c+1][0]
	new_y = contours3[index_c+1][1]
	new_w = contours3[index_c+1][2]
	new_h = contours3[index_c+1][3]
	
	new_w = new_w + ((w + x) - (new_x + new_w))
	new_x = new_x
	new_h = new_h +(new_y - y)
	new_y = y
	remove_index = index_c+1

	return new_x,new_y,new_w,new_h,remove_index

# Preprocessing technique using OTSU
def otsu_preprocess(image):	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	prepimage = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	_, contours, hierarchy = cv2.findContours(prepimage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	return prepimage,contours

def get_ave_area_indie(contours):
	# Doesn't work well if the numbers are not normally distributed. such as "is","initative"
	# Find a way to make this work for all cases
	temp = 0
	for c in contours:
		x, y, w, h = c[0],c[1],c[2],c[3]
		current_area = w*h
		# print("current area: ",current_area)
		temp = temp + current_area
	ave_area = temp/len(contours)
	print("average area: ",ave_area)
	return ave_area

def get_parent(node):
    parent = node.up
    return parent

def backtrack(current_node,fullname):
    print("Backtracking from: ",current_node.name, fullname)
    current_node = get_parent(current_node)
    fullname = fullname[:-1]
    print("After Backtracking: ",current_node.name, fullname)
    return current_node,fullname

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

print("Loading name2node...")
name2node = load_obj("name2node")
# tree = Tree("dictree.nw",format=8)
print("Not gonna make a new tree!")
print("Finished loading name2node!")

def explore_tree(current_node,visited,fullname):
    # exists= True    
    for child in current_node.get_children():
        print("Got this child: ", child.name)

        if(child not in visited):
            visited.append(child)
            if(fullname == "$"):
                print("virgin")
                potential_fullname = child.name
            else:
                potential_fullname = fullname + child.name
                print("Checking potential: ", potential_fullname)
            
            try:
                exists = name2node[potential_fullname]
            except KeyError as okError:
                exists = False
            if(not exists):
                print("Shit doesn't exist in the dict.")
                potential_fullname = fullname
                if(set(current_node.get_children()).issubset(visited)):
                	current_node,fullname = backtrack(current_node,fullname)
            else:
                print("Great! Proceed.")
                current_node = child
                fullname = potential_fullname
                break
           
    return current_node,visited,fullname,exists

def filter_contours(image,contours):
	chosen_contour = []
	print("Length of contours: ",len(contours))
	# cv2.imshow("IMAGE1:  ",image)
	cv2.waitKey(1)
	
	chosen_contour = get_max_area_contour(contours)
	new_img = np.zeros((image.shape), np.uint8)
	
	cv2.drawContours(new_img,[chosen_contour],0,(255,255,255),-1)
	cv2.imshow("newimg",new_img)
	cv2.waitKey(1)
	image = new_img
	
	return image,chosen_contour

# Displays the contours on the input image
def show_contours(contours,image):
	areas=[]
	for c in contours:
		x,y,w,h = cv2.boundingRect(c)
		print(x,y,w,h)
		cv2.rectangle(image, (x, y), (x+w, y+h), (145, 100, 0), 2)
		cv2.imshow("IMAGE",image)
		cv2.waitKey(1)
def feat_extract(image,Y_test):
	data = []
	labels = []
	label = Y_test

	_, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	contours2 = remove_coinsides_letter(contours)
	image = resizeTo20(image)
	centered = transform20x20(image)
	fin = resizeTo28(centered)

	

#####------START OF MACHINE LEARNING AND FEATURE EXTRACTION-------####

	print("Number of ROI: ",len(contours2))


	# Uncomment for error checking if more than 1 roi was detected for recognition
	# Feature extractor must only detect 1 roi
	if(len(contours2) > 1):
		for c in contours2:
			[x, y, w, h] = cv2.boundingRect(c)
			cv2.rectangle(image, (x, y), (x+w, y+h), (145, 100, 0), 4)
			# cv2.imshow("ROI",image)
			print("Your image has more than 1 ROI.")
			sys.exit(0)
		# image,contours = filter_contours(image,contours2)
		# contours2 = []
		# contours2.append(contours)
	
		



	
		# Set parameters for HOG
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
	##Labels contains all the labels of the image
	labels.append((label))
	print("LABEL: ",label)
	return data,labels


# def check(entry):
	
# 	import mmap

# 	f = open('bigger.txt')
# 	s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
# 	b = entry.encode()
# 	if s.find(b) != -1:
# 		return True
# 	else:
# 		return False
def create_potential_tree(words):
	potential_word_tree = Tree()
	pot2node = {}
	for wd in words:

    # If no similar words exist, add it to the base of tree
		target = potential_word_tree

		# Find relatives in the tree
		for pos in range(len(wd), -1, -1):
			# print("pos:",pos)
			root = wd[:pos]
			# print("root: ",root)
			# An existing substring is in the index dict. So you start adding children on that part
			if root in pot2node:
	            # print("if root in name2node: ",name2node[root])
				target = pot2node[root]
				break

	    # Add new nodes as necessary
		fullname = root 
		# print("init fullname: ",fullname)
		# Start on the position kung saan may existing substring OR from the starting letter of the word talaga
		for letter in wd[pos:]:
			# print("Letter: ",letter)
			fullname += letter 
			# print("fullname: ",fullname)
			new_node = target.add_child(name=letter, dist=1.0)
			pot2node[fullname] = new_node
			# print("name2node: ",name2node[fullname])
			target = new_node

	return potential_word_tree
def check(words):
	if(len(words) > 1):
		print("Making potential tree...")
		pot_tree = create_potential_tree(words)
		print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
	else:
		return words[0]
	root = pot_tree.get_tree_root()
	init_node = root
	current_node = init_node
	visited = []
	fullname = "$"
	exit_signal = False
	while((len(fullname)) != len(words[0]) and exit_signal==False):
		print("fullname: ",fullname)
		try:
			current_node,visited,fullname,exists = explore_tree(current_node,visited,fullname) 
		except UnboundLocalError as okError2:
			print("Got an UNBOUNDLOCALERROR")
			fullname = correction(words[0])
			exit_signal = True
			continue
		print("Current Node: ",current_node.name)
		print("Visited nodes: ",visited)
		print("Fullname: ",fullname)

	print("FOUND ANSWER: ", fullname)
	print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
	return fullname
# Calculates the score for all the letters.
def calc_total_score(Ppred,true_labels):
	print("true_labels:", true_labels)
	print("Ppred:", Ppred)
	correct = 0
	print("ALL PREDS: ",Ppred)
	for i,label in enumerate(true_labels):
		# If the code didn't get in here, that means there's a lost data in your predictions
		#
		if label == Ppred[i]: 

			correct = correct + 1
	fscore = correct/len(Ppred)
	print("FINAL SCORE: ", fscore)

def disp_word(Ppred):
	finalString = ''.join(Ppred)
	print(finalString)
	return finalString

def find_proba_of(pred_item,proba_all,set_of_words):
	alphabet = 'abcdefghijklmnopqrstuvwxyz'
	# alphabet = alphabet.upper()
	
	pred_string = np.array_str(pred_item)
	# print("pred_string: ",pred_string)
	# print("length:",len(pred_string))
	label_index = alphabet.index(pred_string)
	print("label_index: ",label_index)
	prob = proba_all[0][label_index]
	# print("proba: ",prob)
	print("proba of {} : {} " .format(pred_string,prob))
	if(prob < 0.60):
		possibles = sorted(zip(proba_all[0], alphabet), reverse=True)[:3]
		set_of_words.append(possibles)
	else:
		possibles = sorted(zip(proba_all[0], alphabet), reverse=True)[:1]
		set_of_words.append(possibles)
 
	return set_of_words

		# largest_integers = heapq.nlargest(3, proba_all[0].tolist())
		# print("largest integers: ",largest_integers )


def process_possible_words(Ppred,true_labels,set_of_words):
	groupLetters = []
	for i in set_of_words:
		letters = []
		print("i: ",i )
		for j in i:
			print("j: ",j[1])
			letters.append(j[1])
		groupLetters.append(letters)
	print("GROUP: ",groupLetters)
	# calc_total_score(Ppred,true_labels)
	final_word = disp_word(Ppred)
	final_word = final_word.lower()
	print(correction(final_word))
	words=[]
	products = it.product(*groupLetters)
	print("PRODUCTS: ",products)
	for prod in products:
		print(prod)
		word = ''.join(prod)
		words.append(word)
	words = [x.lower() for x in words]
	print("WORDS: ",words)
	final_word = check(words)
	return final_word
	             
def upper_and_list(final_word_list):
	new_word_list = []
	for word in final_word_list:
		word = word.upper()
		word = list(word)
		new_word_list.append(word)
	return new_word_list
def score_all_words(true_labels_list,final_word_list):
	
	semis =[]
	final_word_list = upper_and_list(final_word_list)
	# print("TRUE LABELS LIST:",true_labels_list)
	# print("FINAL WORDS LIST: ",final_word_list)
	         
	out_i = 0
                      
	while(out_i < len(true_labels_list)):
		in_i = 0
		score=0                  
		while(in_i<len(true_labels_list[out_i])):
			if(true_labels_list[out_i][in_i] == final_word_list[out_i][in_i]):
				score = score+1
			in_i = in_i +1             
		semifinal_score = score/len(true_labels_list[out_i])                         
		out_i = out_i + 1
		               
		# print("semifinal_score: ",semifinal_score)           
		semis.append(semifinal_score)
	print("ALL SEMIS: ",semis)
	final_score = ((sum(semis))/len(semis)) 
	print("Final score: ",final_score)

def use_svm():
	# loaded_model = joblib.load("lincy_big_svm.pkl")
	loaded_model = joblib.load("lincy_print_small.pkl")
	return loaded_model
def use_dt():
	loaded_model = joblib.load("dt_big_print.pkl")
	return loaded_model
            


def word_recognition(global_ave_area,image,true_labels):
	fword=[]
	allLabels=[]
	# for file in glob.glob(folder_path + '/' + '*.jpg'):
	Ppred = []
             	
	set_of_words=[]
	# x = ImageName(file) # Change the file name
	# image_filename = str(x)
	# image = cv2.imread(image_filename)
	# image_string = image_filename.split('\\')[1]
	# image_string = image_string.split('.')[0]

	# true_labels = list(image_string)
	# allLabels.append(true_labels)
	
	# Resize image so it'll fit on screen
	h,w = image.shape[:2]
	ar = w / h
	nw = 600 
	nh = int(nw / ar)
	nimage = cv2.resize(image,(nw,nh))
	

	#--------Preprocessing sarts here ------#
	prepimage,contours = otsu_preprocess(image)

	contours1 = remove_noise(contours)
	contours2 = remove_coinsides(contours1)
	

	# print("Length of contours2: ",len(contours2))
	# show_contours(contours2,image)
	# kernel = np.ones((3,3),np.uint8)
	# erosion = cv2.erode(prepimage,kernel,iterations = 3)
	# dilation = cv2.dilate(erosion,kernel,iterations = 2)
	# cv2.imshow("prepimage",dilation)
	cv2.imshow("prepimage",prepimage)
	cv2.waitKey(1)

	
	
	# show_contours(contours2,image)
	image = prepimage

	# Get the corresponding x,y,w,h points of each contour
	# This is necessary for sorting the letters L-R
	contours3 = [cv2.boundingRect(c) for c in contours2]

	# Contours are sorted L-R for proper prediction results
	contours3 = sort_LR(contours3)




	i=0
	print(len(contours3))    
	print("pre-contours3: ",contours3)   
	# sys.exit(0)
	# ave_area_indie = get_ave_area_indie(contours3)
	ave_area_indie = global_ave_area
	print("true labels length: ",len(true_labels))
	print("precontours3 length: ",len(contours3))
	# Meaning may letter i or j. Group the dot and body.
	while(len(contours3) > len(true_labels)):
		prev_flag = False
		next_flag = False
		for index_c,cont in enumerate(contours3):

			x,y,w,h =  cont[0],cont[1],cont[2],cont[3]
			if(w*h < (0.1*ave_area_indie)): # Check if the dot is close to the previous or next contour
				print("my average: ",w*h)
				print("That contour is: ",cont)
				prev_diff = abs(contours3[index_c-1][0] - x) 
				next_diff = abs(contours3[index_c+1][0] - x)
				print(prev_diff)
				print(next_diff)
				print("ans:",prev_diff - next_diff)

				if(prev_diff < next_diff): # mas malapit si prev diff kay dot
					print("Closer to prev contour")
					new_x,new_y,new_w,new_h,remove_index = closer_to_prev(contours3,index_c,x,y,w,h)
					prev_flag = True
					break
				else:
					print("closer to next contour")
					new_x,new_y,new_w,new_h,remove_index = closer_to_next(contours3,index_c,x,y,w,h)
					next_flag = True
					break
		# If potential body contour we had in first loop was wrong. correct it.
		if (new_x < 0 or new_y < 0 or new_w < 1 or new_h < 1):
			if(next_flag  == True):
				new_x,new_y,new_w,new_h,remove_index = closer_to_prev(contours3,index_c,x,y,w,h)
			elif (prev_flag == True):
				new_x,new_y,new_w,new_h,remove_index = closer_to_next(contours3,index_c,x,y,w,h)
			else:
				continue
		contours3[remove_index] = new_x,new_y,new_w,new_h

		# except UnboundLocalError as nonMeanError:
			# continue
		del contours3[index_c] 
	print("modified contours3: ",contours3)
	print("contours3 contents: ",contours3)
	print("contours2 length: ",len(contours2))
	for c in contours3: 
		if(i < len(contours3)):
		
			x,y,w,h = c[0],c[1],c[2],c[3]
			# print(x,y,w,h)
			# cv2.rectangle(image, (x, y), (x+w, y+h), (145, 100, 0), 2)
			letter = prepimage[y:y+h,x:x+w]
			letter = cv2.threshold(letter, 0, 255,
				cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
			print("Letter coords: ",x,y,w,h)
			cv2.imshow("letter",letter)
			# # letter,contours2 = filter_contours(letter,contours2)
			# cv2.waitKey(0)
			# path = "C:/Users/User/AppData/Local/Programs/Python/Python36-32/folder1/"
			# cv2.imwrite(path + str(i) + ".jpg",letter)
			print("true labels: ",true_labels)
			print("i: ",i)
			print("true labels in feat extract: ",true_labels[i])
			X_test,Y_test = feat_extract(letter,true_labels[i])
			print("YTEST: ",Y_test)
			# Transform to numpy array
			X_test = np.squeeze(X_test)
			Y_test = np.squeeze(Y_test)
			
			# Reshape to 1,-1 because there is only a single sample
			X_test = X_test.reshape(1,-1)
			Y_test = Y_test.reshape(1,-1)
	
			# Load the model we previously trained
			# This uses the joblib library from sklearn

			loaded_model = use_svm()
			# loaded_model = use_dt()
			# print("num of X_train: ",len(X_train))          
			pred = loaded_model.predict(X_test)
			result = loaded_model.score(X_test, Y_test)
			proba = loaded_model.predict_proba(X_test)
			# print("pred: ",pred)
			# print("result:",result)
			# print("proba: ",proba)

			set_of_words = find_proba_of(pred[0],proba,set_of_words)
			print("set_of_words: ",set_of_words)
         

			# Compile every single recognition to a list
			Ppred.append(pred.tolist()[0])
			
			cv2.imshow("letter",letter)
			cv2.waitKey(1)
			# sys.exit(0)
		i = i + 1                      
	final_word = process_possible_words(Ppred,true_labels,set_of_words)
	fword.append(final_word)
		
				                                          

              
	print("Final word is: ",final_word)
	return final_word
	# score_all_words(allLabels,fword)
	# insert .lower function so we can compare it with the spellchec ker
		
	cv2.waitKey(1)
	cv2.destroyAllWindows()                     
	# exit(0) 
if __name__ == '__main__':
	# x = ImageName(file) # Change the file name
	# image_filename = str(x)
	# image = cv2.imread(image_filename)
	# image_string = image_filename.split('\\')[1]
	# image_string = image_string.split('.')[0]

	# true_labels = list(image_string)
	# allLabels.append(true_labels)
	word_recognition("temp")