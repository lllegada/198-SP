import cv2
import numpy as np 	
import math

def coinside_contour(contours,indices):
	'''stores values of the contours that are co-inside the contour'''
	for c in contours:
		[x, y, w, h] = cv2.boundingRect(c)
		for index,cn in enumerate(contours):
			[i,j,k,l] = cv2.boundingRect(cn)
			if ((i < (x+w) and (i > x)) and ((i+k) < (x+w) and (i+k) > x )):
				
				if((((j+l) < (y+h)) and ((j+l) > y )) and ((j < (y+h)) and (j > y))):
					indices.append(index)
					
				# contours.remove(cn)

	# return indices
	return indices

def isolate_contours(contours,indices,contours2):
	'''Removes the contour found inside a contour'''
	contours2 = [c for i,c in enumerate(contours) if i not in indices]
	return contours2

def get_differences(contours2,differences):
	lastIndex = len(contours2) - 1
	#Checking the biggest space
	for index,c1 in enumerate(contours2): #store differences of each sorted contour
		if index == lastIndex: continue
		[x, y, w, h] = cv2.boundingRect(contours2[index])
		[i, j, k, l] = cv2.boundingRect(contours2[index+1])
		
		dif = (i-(x+w)); #upper-left point of the next contour minus the upper-right point of the current contour
		differences.append(dif)
	return differences

def get_localcut(sliding,localCut,differences):
	for idx, d in enumerate(differences):
		if len(sliding) < 3:  
			sliding.append(d)
		else:
			if (math.ceil(sum(sliding)/len(sliding))) < differences[idx]:
				localCut.append(idx)
			else: 
				del sliding[0]
				sliding.append(d)	
	return sliding,localCut

def get_globalcut(differences,globalCut):
	max_val = max(differences)

	for index, d in enumerate(differences):
		cur = (d / max_val) * 100
		if (cur > 50):
			globalCut.append(index)
	return globalCut

def check_rare(globalCut,localCut):
	rare = [val for i,val in enumerate(globalCut) if val not in localCut]
	return rare

def compute_final_intersection(differences,globalCut,intersection):
	'''Use this only if there is a rare case'''
	end_max = len(differences) - 1
	for d in globalCut:
		if(d<3):
				if(d == 0):
					avg_start = 0
				else:
					start=range(0,(d-1)+1)
		else:
			start=range(d-3,(d-1)+1)
		if(d!=0):
			avg_start = math.ceil(sum(differences[start[0]:start[1] + 1])/len(start))

		if(d>= (len(differences)-3)):
			if d == len(differences):
				avg_end=0
			else:
				end=range(d+1,end_max)
		else:
			end=range(d+1,(d+3)+1)

		if(d!=len(differences)-1):
			avg_end = math.ceil(sum(differences[end[0]:end[1] + 1])/len(end))

		if((avg_start < differences[d]) and (avg_end < differences[d])):
			intersection.append(d)
	return intersection
def segment_words(contours2,globalCut):
	x1,y1 = None,None
	i=0
	while i < len(contours2)-1:
		c = cv2.boundingRect(contours2[i])
		d = cv2.boundingRect(contours2[i+1])
		if(x1 is None):
			x1 = c[0]
		if(y1 is None):
			y1 = c[1]
	#adjust height of the y1 to align with the height of the taller contour
		if(y1 > d[1]):
			y1 = d[1]
		
		if(i in globalCut):
			x2 = c[0] + c[2] #x+w
			y2 = c[1] + c[3] #y+h
			box = [x1,y1,x2,y2]
			words.append(box)
			x1 = d[0]
			y1 = d[1]
			x2 = None
			y2 = None
		i=i+1
	if(i == len(contours2)-1): #if lagpas na sya sa last 2 element, then consider it as out of bounds
		x2 = d[0] + d[2] #x+w
		y2 = d[1] + d[3] #y+h
		box = [x1,y1,x2,y2]
		words.append(box)

	return contours2,box,words


def draw_character_roi(contours2):
	for c in contours2:
		[x, y, w, h] = cv2.boundingRect(c)
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
def draw_word_roi(box,words):
	for box in words:
		cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
def resize_image(image):
#Resizing the image to be shown
	h,w = image.shape[:2]
	ar = w / h
	nw = 1300
	nh = int(nw / ar)
	nimage = cv2.resize(image,(nw,nh))
	return nimage
def sort_contours(contours2):
	boundingBoxes = [cv2.boundingRect(c) for c in contours2]
	(contours2, boundingBoxes) = zip(*sorted(zip(contours2, boundingBoxes),
	key=lambda b:b[1][0], reverse=False))
	return contours2,boundingBoxes

image = cv2.imread("pic10.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
denoise = cv2.fastNlMeansDenoising(gray,None,10,7,21)
blur = cv2.GaussianBlur(denoise, (5, 5), 0)
ret, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
indices=[]
contours2=[]
differences=[]

sliding = []
localCut = []
globalCut = [] #stores indices of differences that has 50% chance of being a space
words = []
box=[]
coinside_contour(contours,indices)
contours2 = isolate_contours(contours,indices,contours2)


<<<<<<< HEAD
# SORT contours2 according t the x values in increasing order(largest pixel value at the last)
contours2,boundingBoxes = sort_contours(contours2)

differences = get_differences(contours2,differences)
sliding,localCut = get_localcut(sliding,localCut,differences)
globalCut = get_globalcut(differences,globalCut)

# Check for the intersection of localcut and globalcut values(INDICES)
intersection = [overlap for i, overlap in enumerate(globalCut) if overlap in localCut]

# Keep values that are not in the intersection
globalCut = [cut for i, cut in enumerate(globalCut) if i not in intersection ]

#rare value = value present in global cut but not in local cut
#Checks if there is a rare value
if check_rare(globalCut,localCut): #0 = false ; others = true
	intersection = compute_final_intersection(differences,globalCut,intersection)

#segment into words
contours2,box,words = segment_words(contours2,words)

draw_character_roi(contours2) #rectangles by character
draw_word_roi(box,words) #rectangles by words
nimage = resize_image(image)

cv2.imshow("RESULT",nimage)
cv2.waitKey(0)
cv2.destroyAllWindows()