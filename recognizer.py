'''
Problems:
	letter i and j must be contoured as one box
	difference between two adjacent contours looks off

'''


import cv2
import numpy as np 	
import math


image = cv2.imread("pic1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
denoise = cv2.fastNlMeansDenoising(gray,None,10,7,21)
blur = cv2.GaussianBlur(denoise, (5, 5), 0)

ret, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)


_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

indices=[]
for c in contours:
	[x, y, w, h] = cv2.boundingRect(c)
	

	
	
	for index,cn in enumerate(contours):
		
		[i,j,k,l] = cv2.boundingRect(cn)
		
		if ((i < (x+w) and (i > x)) and ((i+k) < (x+w) and (i+k) > x )):
			if((((j+l) < (y+h)) and ((j+l) > y )) and ((j < (y+h)) and (j > y))):
		
				indices.append(index)
				# contours.remove(cn)

print ('1st: Number of contours are: %d -> ' %len(contours))
contours2 =[]
contours2 = [c for i,c in enumerate(contours) if i not in indices]

# SORT contours2 according t the x values in increasing order(largest pixel value at the last)
# Determine the maximum probable case that it is a space between words
boundingBoxes = [cv2.boundingRect(c) for c in contours2]
(contours2, boundingBoxes) = zip(*sorted(zip(contours2, boundingBoxes),
key=lambda b:b[1][0], reverse=False))

differences=[]

lastIndex = len(contours2) - 1
#Checking the biggest space
for index,c1 in enumerate(contours2):
	# print ('SIZE: %d' %(len(contours2)))
	# if(index < (len(contours2))-1):
	if index == lastIndex: continue
	[x, y, w, h] = cv2.boundingRect(contours2[index])
	[i, j, k, l] = cv2.boundingRect(contours2[index+1])
	# print ('x: %d \n y:%d \n x+w:%d \n y+h:%d' %(x,y,x+w,y+h))
	# print ('i: %d \n j:%d \n i+k:%d \n j+l:%d' %(i,j,i+k,j+l))
	dif = (i-(x+w));
	differences.append(dif)

sliding = []
localCut = []

for idx, d in enumerate(differences):
		if len(sliding) < 3:  
			sliding.append(d)
		else:
			if (math.ceil(sum(sliding)/len(sliding))) < differences[idx]:
				localCut.append(idx)
			else: 
				del sliding[0]
				sliding.append(d)

print("localcut values are:")
print(localCut)

max_val = max(differences)
globalCut = []

for index, d in enumerate(differences):
	cur = (d / max_val) * 100
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
	for d in globalCut:
# 		sum_start = 0
# 		sum_end = 0
# 		# difD = differences[d]
# 		# print("<!----START OF LOOPING---!>")
		# if d < 3:
# 			if d == 0:
# 				break
# 			else:
# 				start=range(0,(d-1)+1)
# 		else:
# 			start=range(d-3,(d-1)+1)

# 		# print("Value of start:")
# 		# print(start)

# 	# gets the sum of start
# 		if(d!=0):
# 			for x in list(start):
# 				# print("\nX:",x)
# 				sum_start += differences[x]
# 				# print("sum start: ",sum_start)

# 			avg_start = math.ceil(sum_start / len(start))
# 			# print("average: ",avg_start)

# 	# end
# 		# difD = differences[d]
# 		if d >= (len(differences)-3) :
# 			if d == len(differences):
# 				end=range(end_max,end_max+1)
# 			else:
# 				end=range(d+1,end_max)
# 		else:
# 			end=range(d+1,(d+3)+1)

# 		# print("Value of end:")
# 		# print(end)

# 		if(end != len(end)-1):
# 			for x in list(end):
# 				# print("\nX:",x)
# 				sum_end += differences[x]
# 				# print("sum end: ",sum_end)

# 			avg_end = math.ceil(sum_end / len(end))
# 			# print("average: ",avg_end)

# 		# compare if the average of start and end is less than the d/midpoint, d is a space
		if(d<3):
				if(d == 0):
					avg_start = 0
				else:
					start=range(0,(d-1)+1)
					# if d < 3:
		else:
			start=range(d-3,(d-1)+1)
		if(d!=0):
			avg_start = math.ceil(sum(differences[start[0]:start[1] + 1])/len(start))



		print("d is:",d)
		if(d>= (len(differences)-3)):
			if d == len(differences):
				avg_end=0
			else:
				end=range(d+1,end_max)
		else:
			end=range(d+1,(d+3)+1)

		print("end is:",end)
		if(d!=len(differences)-1):
			avg_end = math.ceil(sum(differences[end[0]:end[1] + 1])/len(end))

		if((avg_start < differences[d]) and (avg_end < differences[d])):
			intersection.append(d)

print("\n\nIntersection: ",intersection)	

# get the values of start and end then find the avg


# for x, val in enumerate(globalCut):
# 	if ((sum(differences[start])/len(start)) < differences[val]):

#<--------NEXT------->

words = []
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
		
	print("differences[i]")
	print(differences[i])
	print("i:")
	print(i)

	if(i in globalCut): #pakicheck if the index overflows
		print("Pumasok si i:")
		print(i)
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

print("length of contours2", len(contours2))
print("length of differences", len(differences))



for c in contours2:
	[x, y, w, h] = cv2.boundingRect(c)
	cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
for box in words:
	# [x, y, w, h] = cv2.boundingRect(c)
	cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

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