# usr/bin/python -tt

import numpy as np
import cv2, uuid, os


def segment(image):
	fixed_size_image = cv2.resize(image, (200,180))
	gray_image = cv2.cvtColor(fixed_size_image, cv2.COLOR_BGR2GRAY)
	blur_image = cv2.GaussianBlur(gray_image, (5,5), 0)
	__, binary_image = cv2.threshold(blur_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
	binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
	__, contours, __ = cv2.findContours(binary_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	list_coutours = []
	rois = []
	for c in contours:
		x, y, w, h = cv2.boundingRect(c)
		area = w*h
		if area >= 1000 and area <= 2500 and h > w:
			list_coutours.append(c)

	a = binary_image[list_coutours[0][0][0][1] + 5][list_coutours[0][0][0][0] + 5]
	if a == 255:
		binary_image = cv2.bitwise_not(binary_image)

	for lc in list_coutours:
		x, y, w, h = cv2.boundingRect(lc)
		rois.append(binary_image[y:y + h, x:x + w])

		#cv2.drawContours(fixedSizeImage, list_coutours, -1, (0,255,0), 2)
		#cv2.imshow("3", fixedSizeImage)


		# sort rois
		
	return np.array(list(map(lambda r: cv2.resize(r, (40, 80), interpolation = cv2.INTER_LINEAR), rois)))
'''
		for r in rois:
			img_scaled = cv2.resize(r,(40, 80), interpolation = cv2.INTER_CUBIC)
			name = str(uuid.uuid4())
			cv2.imshow(name, r)
			cv2.imshow(name + '1', img_scaled)

			#cv2.imwrite("data/" + imageName.split('.')[0] + "-" + name + ".png", r)

		cv2.waitKey(0)
		return rois

'''
