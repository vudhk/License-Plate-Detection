# usr/bin/python -tt

import numpy as np
import cv2, uuid, os

IMG_WIDTH, IMG_HEIGHT = 32, 40

def segment(image):
	if image.shape[1] >= image.shape[0] * 2 :
		f = 90 / image.shape[0]
		fixed_size_image = cv2.resize(image, None, fx=f, fy=f, interpolation = cv2.INTER_LINEAR)
	else:
		f = 180 / image.shape[0]
		fixed_size_image = cv2.resize(image, None, fx=f, fy=f, interpolation = cv2.INTER_LINEAR)

	#cv2.imshow('123', fixed_size_image)
	
	gray_image = cv2.cvtColor(fixed_size_image, cv2.COLOR_BGR2GRAY)
	blur_image = cv2.GaussianBlur(gray_image, (5,5), 0)
	__, binary_image = cv2.threshold(blur_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
	binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
	__, contours, __ = cv2.findContours(binary_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	global bound_rects
	global list_coutours

	bound_rects = []
	list_coutours = []
	rois = []
	for c in contours:
		x, y, w, h = cv2.boundingRect(c)
		area = w*h
		if area >= 600 and area <= 4000 and h/w >= 1.2 and h/w <= 5 and not contours_nested(c, x, y, w, h):
			list_coutours.append(c)
			bound_rects.append((x, y, w, h))

	a = binary_image[list_coutours[0][0][0][1] + 5][list_coutours[0][0][0][0] + 5]
	if a == 255:
		binary_image = cv2.bitwise_not(binary_image)


	# sort list_coutours
	epsilon = 45
	dtype = [('value', int), ('index', int)]
	arr_tmp = np.array([(value[0] + pow(10 if value[1] <= epsilon else 100, 2), idx) for idx, value in enumerate(bound_rects)], dtype=dtype)
	arr_tmp = np.sort(arr_tmp, order='value')
	list_coutours = list(map(lambda value: list_coutours[value[1]], arr_tmp))


	for lc in list_coutours:
		x, y, w, h = cv2.boundingRect(lc)
		rois.append(binary_image[y:y + h, x:x + w])
		#cv2.drawContours(fixed_size_image, list_coutours, -1, (0,255,0), 1)
		#cv2.imshow("3", fixed_size_image)


	#for r in rois:
		#name = str(uuid.uuid4())
		#cv2.imshow(name, r)
		#cv2.imwrite("data/" + imageName.split('.')[0] + "-" + name + ".png", r)

	#cv2.waitKey(0)

	return np.array(list(map(lambda r: left_to_fill(r), rois)))



def contours_nested(c, x, y, w, h):
	arr = list(filter(lambda br: br[0] <= x and br[2] >= x - br[0] and br[1] <= y and br[3] >= y - br[1], bound_rects))
	if len(arr) > 0 :
		return True
	else:
		arr = list(filter(lambda br: br[0] >= x and br[0] - x <= w and br[1] >= y and h >= br[1] - y, bound_rects))
		if len(arr) > 0:
			for elem in arr:
				idx = bound_rects.index(elem)
				bound_rects[idx] = (x, y, w, h)
				list_coutours[idx] = c
		return False

def left_to_fill(image):
	f = IMG_HEIGHT/image.shape[0]
	resize_img = cv2.resize(image, None, fx=f, fy=f, interpolation = cv2.INTER_LINEAR)
	bg = np.uint8(np.full((IMG_HEIGHT, IMG_WIDTH), 255))
	bg[0:resize_img.shape[0], 0:resize_img.shape[1]] = resize_img
	#resize_img = cv2.resize(image, (40, 80), interpolation = cv2.INTER_LINEAR)
	#cv2.imwrite('/home/vudhk/Desktop/License-Plate-Detection/Samples/segnments/{}.jpg'.format(str(uuid.uuid4())), bg)
	return bg


#if __name__ == '__main__':
#	segment(cv2.imread('/home/vudhk/Desktop/License-Plate-Detection/Samples/vietnammc.jpg'))
