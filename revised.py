'''
Problems:
	letter i and j must be contoured as one box
	difference between two adjacent contours looks off

'''


import cv2
import numpy as np 	
import math


image = cv2.imread("hand2.jpg")
h,w = image.shape[:2]
ar = w / h
nw = 1300
nh = int(nw / ar)
image = cv2.resize(image,(nw,nh))





# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# cv2.imshow('hsv', hsv[:,:,0])
# (thresh, im_bw) = cv2.threshold(hsv[:,:,0], 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# cv2.imshow('OTSU', im_bw)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Smooth image
blur = cv2.GaussianBlur(gray, (3, 3), 0)
filtered = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 3)

# Some morphology to clean up image
kernel = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(filtered,cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
kernel2 = np.ones((9,9), np.uint8)
closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2, iterations =1)

cv2.imshow("filtered",filtered)
cv2.imshow("closing",closing)
# dst = cv2.fastNlMeansDenoising(filtered,None,10,7,21)
# cv2.imshow("denoise",dst)
# denoise = cv2.fastNlMeansDenoising(closing,None,10,7,21)
# blur = cv2.GaussianBlur(denoise, (5, 5), 0)
# cv2.imshow("to use",blur)
# ret, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)


_, contours, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
sliding = []
contours2 =[]
localCut = []
differences=[]
globalCut = [] #stores indices of differences that has 50% chance of being a space
words = []


for c in contours:
	x, y, w, h = cv2.boundingRect(c)
	print(x,y,w,h)
	cv2.rectangle(image, (x, y), (x+w, y+h), (145, 100, 0), 2)
cv2.imshow('res',image)
cv2.waitKey(0)
exit(0)
indices=[] #stores values of the contours that are co-inside the contour
for c in contours:
	[x, y, w, h] = cv2.boundingRect(c)
	for index,cn in enumerate(contours):
		
		[i,j,k,l] = cv2.boundingRect(cn)
		
		if ((i < (x+w) and (i > x)) and ((i+k) < (x+w) and (i+k) > x )):
			# print("inside")
			if((((j+l) < (y+h)) and ((j+l) > y )) and ((j < (y+h)) and (j > y))):
		
				indices.append(index)
				# contours.remove(cn)
# print("num of contours",len(contours))
# print("# of indices:",len(indices))
# print ('1st: Number of contours are: %d -> ' %len(contours))

contours2 = [c for i,c in enumerate(contours) if i not in indices]
# print ('2nd: Number of contours are: %d -> ' %len(contours2))


# SORT contours2 according t the x values in increasing order(largest pixel value at the last)
# Determine the maximum probable case that it is a space between words
rect = [cv2.boundingRect(c) for c in contours2]
# sort all rect by their y
rect.sort(key=lambda b: b[1])

# initially the line bottom is set to be the bottom of the first rect
line_bottom = rect[0][1]+rect[0][3]-1
line_begin_idx = 0
p=0
# print("length of rect:",len(rect))
for ix in range(len(rect)):
	# when a new box's top is below current line's bottom
	# it's a new line
	# print("line_bottom:", line_bottom)
	if rect[ix][1] > line_bottom:
		print("Value of i:",ix)
		# sort the previous line by their x
		rect[line_begin_idx:ix] = sorted(rect[line_begin_idx:ix], key=lambda b: b[0])
		contours2 = rect[line_begin_idx:ix]
		# print("c:",rect)
		#stores the differences of each contour from adjacent contour(left->right)
		# for c in contours2:
		# 	x, y, w, h = c
		# 	print("VALUES")
		# 	print(x,y,w,h)
			# cv2.rectangle(image, (x, y), (x+w, y+h), (20, 25, 0), 2)
			
		lastIndex = len(contours2) - 1
		#Checking the biggest space
		for index,c1 in enumerate(contours2):
			# print ('SIZE: %d' %(len(contours2)))
			# if(index < (len(contours2))-1):
			if index == lastIndex: continue
			[x, y, w, h] = contours2[index]
			[i, j, k, l] = contours2[index+1]
			# print ('x: %d \n y:%d \n x+w:%d \n y+h:%d' %(x,y,x+w,y+h))
			# print ('i: %d \n j:%d \n i+k:%d \n j+l:%d' %(i,j,i+k,j+l))
			dif = abs((i-(x+w))); #upper-left point of the next contour minus the upper-right point of the current contour
			print("dif: ", dif)
			differences.append(dif)


		print("length of differences:",len(differences))
		print("differences:",differences)



		for idx, d in enumerate(differences):
				if len(sliding) < 3:  
					sliding.append(d)
					if len(differences) < 3: #max is 3 words
						# print("i:",idx)
						localCut.append(idx)
				else:
					# print("else  i:",idx)
					if (math.ceil(sum(sliding)/len(sliding))) < differences[idx]:
						localCut.append(idx)
					else: 
						del sliding[0]
						sliding.append(d)

		print("localcut values are:")
		print(localCut)
		# if(len(localCut)!=0):
		max_val = max(differences)

		for index, d in enumerate(differences):
			cur = (d / max_val) * 100
			# print("cur:",cur)
			if (cur > 50):
				globalCut.append(index)

		print("global cut: ")
		print(globalCut)

		intersection = [overlap for i, overlap in enumerate(globalCut) if overlap in localCut]

		globalCut = [cut for i, cut in enumerate(globalCut) if i not in intersection ]


		print("Intersection values", intersection)
		print("new global cut: ", globalCut)
		#If space threshold found, create words array and and get the points of the 
		#DRAWING
		rare = [val for i,val in enumerate(globalCut) if val not in localCut]
		print("RARE:", rare)
		if rare:
		# 	start =[]
		# 	end = []
			end_max = len(differences) - 1
		# 	# start
			for d in rare:
				# compare if the average of start and end is less than the d/midpoint, d is a space
				print("d: ",d)
				if(d<3):
						if(d == 0) or (d == 1):
						# if(d==0):
							avg_start = 0
						else:
							start=range(0,d)
							# if d < 3:
				else:
					start=range(d-3,(d-1)+1)
				if(d > 1):
				# if(d != 0):
					avg_start = math.ceil(sum(differences[start[0]:start[1] + 1])/len(start))
					
					
				print("avg_start: ",avg_start)


				# print("d is:",d)
				end_max = len(differences)-1

				if(d>= (len(differences)-3)):
					if d == end_max or d == end_max - 1:
						avg_end=differences[end_max-1]
					else:
						end=range(d+1,end_max+1)
				else:
					end=range(d+1,(d+3)+1)

				
				# if(d!=len(differences)-1):
				if(d < end_max-1):
					# print("d: ",d)
					print("end is:",end)
					print("length of diff:",len(differences))
					print("diff[end[0]]:", end[0])
					print("diff[end[1]]:", end[1])
					print("length of end: ", len(end))

					avg_end = math.ceil(sum(differences[end[0]:end[1] + 1])/len(end))
				print("avg_end: ", avg_end)

				if((avg_start <= differences[d]) and (avg_end <= differences[d])):
					intersection.append(d)

		print("\n\nIntersection: ",intersection)	
		print("differences: ",differences)
	# get the values of start and end then find the avg
	# else:
	# 	intersection = localCut;

		# for x, val in enumerate(globalCut):
		# 	if ((sum(differences[start])/len(start)) < differences[val]):

		#<--------NEXT------->

		x1,y1 = None,None
		i=0
		while i < len(contours2)-1:
			c = contours2[i]
			d = contours2[i+1]
			if(x1 is None):
				x1 = c[0]
			if(y1 is None):
				y1 = c[1]
		#adjust height of the y1 to align with the height of the taller contour
			if(y1 > d[1]): 
				y1 = d[1] #y1 is the the y-coordinate stored in words array

				
			# print("differences[i]")
			# print(differences[i])
			# print("i:")
			# print(i)

		#if contour[i+1][y+h] > contour[i][y+h]
			
			#ang index ara sa final cuts
			if(i in intersection): #pakicheck if the index overflows
				# print("Pumasok si i:")
				# print(i)
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
		# print("boxes:",words)
		# print("length of contours2", len(contours2))
		# print("length of differences", len(differences))
		# print("WORDS:",words)


		# for c in contours2:
		# 	[x, y, w, h] = c
		# 	cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

		for box in words:
			# [x, y, w, h] = cv2.boundingRect(c)
			cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (90, p , 80), 2)
		

		line_begin_idx = ix

		# print("P: ",p)
	p = p + 60
	words = []
	differences=[]
	sliding=[]
	localCut=[]
	globalCut=[]
	intersection=[]
	contours2=[]
	# regardless if it's a new line or not
	# always update the line bottom
	# line_bottom = max(rect[i][1]+rect[i][3]-1, line_bottom)
	line_bottom = (rect[ix][1]+rect[ix][3]-1)
# sort the last line
rect[line_begin_idx:] = sorted(rect[line_begin_idx:], key=lambda b: b[0])
contours2 = rect[line_begin_idx:]
# print("line_begin_idx:", line_begin_idx)
print("NEW")
# for c in contours2:
# 	x, y, w, h = c
# 	print("VALUES")
# 	print(x,y,w,h)
	# cv2.rectangle(image, (x, y), (x+w, y+h), (20, 25, 0), 2)
	
lastIndex = len(contours2) - 1
#Checking the biggest space
for index,c1 in enumerate(contours2):
	print ('SIZE: %d' %(len(contours2)))
	# if(index < (len(contours2))-1):
	if index == lastIndex: continue
	[x, y, w, h] = contours2[index]
	[i, j, k, l] = contours2[index+1]
	# print ('x: %d \n y:%d \n x+w:%d \n y+h:%d' %(x,y,x+w,y+h))
	# print ('i: %d \n j:%d \n i+k:%d \n j+l:%d' %(i,j,i+k,j+l))
	dif = abs((i-(x+w))); #upper-left point of the next contour minus the upper-right point of the current contour
	differences.append(dif)


print("length of differences:",len(differences))
print("differences:",differences)



for idx, d in enumerate(differences):
		if len(sliding) < 3:  
			sliding.append(d)
			if len(differences) < 3: #max is 3 words
				# print("i:",idx)
				localCut.append(idx)
		else:
			# print("else  i:",idx)
			if (math.ceil(sum(sliding)/len(sliding))) < differences[idx]:
				localCut.append(idx)
			else: 
				del sliding[0]
				sliding.append(d)

print("localcut values are:")
print(localCut)
if(len(localCut)!=0):
	max_val = max(differences)

	for index, d in enumerate(differences):
		cur = (d / max_val) * 100
		# print("cur:",cur)
		if (cur > 50):
			globalCut.append(index)

	print("global cut: ")
	print(globalCut)

	intersection = [overlap for i, overlap in enumerate(globalCut) if overlap in localCut]

	globalCut = [cut for i, cut in enumerate(globalCut) if i not in intersection ]


	print("Intersection values", intersection)
	print("new global cut: ", globalCut)
	#If space threshold found, create words array and and get the points of the 
	#DRAWING
	rare = [val for i,val in enumerate(globalCut) if val not in localCut]
	print("RARE:", rare)
	if rare:
	# 	start =[]
	# 	end = []
		end_max = len(differences) - 1
	# 	# start
		for d in rare:
			# print("curD:",d)
			# compare if the average of start and end is less than the d/midpoint, d is a space
			if(d<3):
				if(d == 0) or (d == 1):
				# if(d==0):
					avg_start = 0
				else:
					start=range(0,d)
					# if d < 3:
			else:
				start=range(d-3,(d-1)+1)
			if(d > 1):
			# if(d != 0):
				avg_start = math.ceil(sum(differences[start[0]:start[1] + 1])/len(start))


			# print("d is:",d)
			# end_max = len(differences)-1
			if(d>= (len(differences)-3)):
				if d == end_max or d == end_max - 1:
					avg_end=differences[end_max-1]
				else:
					end=range(d+1,end_max+1)
			else:
				end=range(d+1,(d+3)+1)

			
			# if(d!=len(differences)-1):
			if(d < end_max-1):
				print("d: ",d)
				print("end is:",end)
				print("length of diff:",len(differences))
				print("diff[end[1]]:", end[1])
				avg_end = math.ceil(sum(differences[end[0]:end[1] + 1])/len(end))

			if((avg_start <= differences[d]) and (avg_end <= differences[d])):
				intersection.append(d)

	print("\n\nIntersection: ",intersection)	
else:
	# intersection = range(0,len(differences))
	intersection = range(0,len(differences))
# get the values of start and end then find the avg


# for x, val in enumerate(globalCut):
# 	if ((sum(differences[start])/len(start)) < differences[val]):

#<--------NEXT------->

x1,y1 = None,None
i=0
while i < len(contours2)-1:
	c = contours2[i]
	d = contours2[i+1]
	if(x1 is None):
		x1 = c[0]
	if(y1 is None):
		y1 = c[1]
#adjust height of the y1 to align with the height of the taller contour
	if(y1 > d[1]): 
		y1 = d[1] #y1 is the the y-coordinate stored in words array


		
	# print("differences[i]")
	# print(differences[i])
	# print("i:")
	# print(i)

#if contour[i+1][y+h] > contour[i][y+h]
	
	#ang index ara sa final cuts
	if(i in intersection): #pakicheck if the index overflows
		# print("Pumasok si i:")
		# print(i)
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
# print("boxes:",words)
# print("length of contours2", len(contours2))
# print("length of differences", len(differences))
# print("WORDS:",words)


# for c in contours2:
# 	[x, y, w, h] = c
# 	cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

for box in words:
	# [x, y, w, h] = cv2.boundingRect(c)
	cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (90, p , 80), 2)



# print("P: ",p)
#Resizing the image to be shown
h,w = image.shape[:2]
ar = w / h
nw = 1300
nh = int(nw / ar)
nimage = cv2.resize(image,(nw,nh))

# print ('Number of contours are: %d -> ' %len(contours))

#Checks if naka arrange in ascending order
# [x, y, w, h] = cv2.boundingRect(contours2[1])
# cv2.rectangle(image, (x, y), (x + w, y + h), (255,0,0), 2)

cv2.imshow("RESULT",nimage)
# cv2.imwrite('output.jpg',image)
cv2.waitKey(0)
cv2.destroyAllWindows()