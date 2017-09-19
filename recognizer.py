'''
Problems:
*needs better noise reduction than blurring.
*considers the area within a character as a contour
'''


import cv2
import numpy as np 


image = cv2.imread("hello.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
denoise = cv2.fastNlMeansDenoising(gray,None,10,7,21)
blur = cv2.GaussianBlur(denoise, (5, 5), 0)

ret, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)


_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# print "DONE: %f DIVIDED BY %f EQUALS %f, SWEET MATH BRO!" % (first, second, ans)
indices=[]
for c in contours:
	[x, y, w, h] = cv2.boundingRect(c)
	print('C')
	# print(c)
	print ('x: %d \n y: %d \n x+w: %d \n y+h: %d ' %(x,y,x+w,y+h))
	
	for index,cn in enumerate(contours):
		print('NEW CONTOUR')
		# i (x) - vertex points on the upper - left
		# j (y) - vertex points or the lower - right
		[i,j,k,l] = cv2.boundingRect(cn)
		# print(cn)
		# print ('x: %d \n y: %d \n x+w: %d \n y+h: %d \n i: %d \n j:%d \n i+k:%d \n l:%d' %(x,y,x+w,y+h,i,j,i+k,l))
		print ('i: %d \n j:%d \n i+k:%d \n j+l:%d' %(i,j,i+k,j+l))
		# print 'x:%d '%(x)
		if ((i < (x+w) and (i > x)) and ((i+k) < (x+w) and (i+k) > x )):
			if((((j+l) < (y+h)) and ((j+l) > y )) and ((j < (y+h)) and (j > y))):
		# 	print ('i: %d \n j:%d \n i+k:%d \n l:%d' %(i,j,i+k,l))
				# cv2.rectangle(image, (i, j), (i + k, j + l), (0, 0, 255), 2)
				# 
				indices.append(index)
				# contours.remove(cn)

print ('1st: Number of contours are: %d -> ' %len(contours))
contours2 =[]
contours2 = [c for i,c in enumerate(contours) if i not in indices]
# for i in indices:
# 	print("indices")
# 	print(i)
# 	for ci,c in enumerate(contours):
# 		print ('2nd: Number of contours are: %d -> ' %len(contours))
# 		print("CI")
# 		print(ci)
# 	 if len(indices) != 0:
# 	 	del contours[i]
	
# 		if(i != ci):
# 			contours2.append(c)
			

for c in contours2:
	[x, y, w, h] = cv2.boundingRect(c)
	cv2.rectangle(image, (x, y), (x + w, y + h), (130, 0, 255), 2)

print ('Number of contours are: %d -> ' %len(contours))

cv2.imshow("RESULT",image)
cv2.waitKey(0)
cv2.destroyAllWindows()