import numpy as np
import cv2, uuid, os

def processing(file_name):
	img = cv2.imread(file_name, 0)
	return [img]

'''
imageNames = os.listdir("Samples")

for imageName in imageNames:
	#imageName = "vietnam.jpg"
	originImage = cv2.imread("Samples/" + imageName)


	fixedSizeImage = cv2.resize(originImage, (200,180))
	#cv2.imshow("1", fixedSizeImage)

	grayImage = cv2.cvtColor(fixedSizeImage, cv2.COLOR_BGR2GRAY)
	blurImage = cv2.GaussianBlur(grayImage, (5,5), 0)

	_, binaryImage = cv2.threshold(blurImage, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

	#cv2.imshow("2", binaryImage)

	cntImage = binaryImage.copy()

	__, contours, hierarchy = cv2.findContours(cntImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	listCoutours = []
	rois = []


	for c in contours:
		x, y, w, h = cv2.boundingRect(c)
		area = w*h
		if area >= 1000 and area <= 2500 and h > w:
			listCoutours.append(c)


	a = binaryImage[listCoutours[0][0][0][1] + 5][listCoutours[0][0][0][0] + 5]
	#print(str(a) + " --- " + imageName)
	if a == 255:
		binaryImage = cv2.bitwise_not(binaryImage)

	for lc in listCoutours:
		x, y, w, h = cv2.boundingRect(lc)
		rois.append(binaryImage[y:y + h, x:x + w])


	#cv2.drawContours(fixedSizeImage, listCoutours, -1, (0,255,0), 2)
	#cv2.imshow("3", fixedSizeImage)

	for r in rois:
		img_scaled = cv2.resize(img,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_LINEAR)
		name = str(uuid.uuid4())
		#cv2.imshow(name, r)
		cv2.imwrite("data/" + imageName.split('.')[0] + "-" + name + ".png", r)

	#break

cv2.waitKey(0)
'''


