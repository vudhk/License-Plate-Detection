# usr/bin/python -tt

import numpy as np
import cv2, uuid, os
import glob
import shutil
from math import floor

IMG_WIDTH, IMG_HEIGHT = 40, 40

origin_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
img_processing_output = origin_folder + "/ImgProcessOutput"
if not os.path.exists(img_processing_output):
	os.makedirs(img_processing_output)
else:
	files = glob.glob(img_processing_output + '/*')
	for f in files:
	    os.remove(f)

def segment(image):
	global fixed_size_image
	if image.shape[1] >= image.shape[0] * 2 :
		f = 90 / image.shape[0]
		fixed_size_image = cv2.resize(image, None, fx=f, fy=f, interpolation = cv2.INTER_LINEAR)
	else:
		f = 180 / image.shape[0]
		fixed_size_image = cv2.resize(image, None, fx=f, fy=f, interpolation = cv2.INTER_LINEAR)

	fixed_size_image = cv2.fastNlMeansDenoisingColored(fixed_size_image,None,10,10,7,21)
	cv2.imwrite(img_processing_output + '/1-fixed_size_image.png', fixed_size_image)
	global binary_image
	
	gray_image = cv2.cvtColor(fixed_size_image, cv2.COLOR_BGR2GRAY)

	binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
	cv2.imwrite(img_processing_output + '/2.1-binary_image.png', binary_image)

	#binary_image  = cv2.morphologyEx(binary_image,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)));
	#cv2.imwrite(img_processing_output + '/2.2-binary_image_MORPH_OPEN.png', binary_image )
 
	binary_image = cv2.morphologyEx(binary_image,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)));
	cv2.imwrite(img_processing_output + '/2.3-binary_image_MORPH_CLOSE.png', binary_image )
	print('-- binary image processed!')

	cv2.rectangle(binary_image, (0,0), (binary_image.shape[1] - 1, binary_image.shape[0] - 1), (255,255,255), 1)
	__, contours, __ = cv2.findContours(binary_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	img_clone = fixed_size_image.copy()
	cv2.drawContours(img_clone, contours, -1, (0,255,0), 1)
	cv2.imwrite(img_processing_output + '/4-bounding_rect_full.png', img_clone)

	img_clone_1 = fixed_size_image.copy()


	list_contours = []
	rois = []
	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt)
		if is_number_region(x, y, w, h, cv2.contourArea(cnt)): 
			list_contours.append(cnt)
			cv2.rectangle(img_clone_1, (x,y), (x+w, y+h), (0,0,255), 2)
			#cv2.drawContours(img_clone_1,[c],0,(0,0,255),2)
	cv2.imwrite(img_processing_output + '/4.5-bounding_rect_full.png', img_clone_1)

	# Tạo mặt nạ
	mask = np.zeros((binary_image.shape[0], binary_image.shape[1], 1), dtype = "uint8")
	margin = 5
	for cnt in list_contours:
		x, y, w, h = cv2.boundingRect(cnt)
		cv2.rectangle(mask, (x + margin,y + margin), (x + w - margin, y + h - margin), (255,255,255), -1)
	
	cv2.imwrite(img_processing_output + '/4.6-mask.png', mask)
	_, list_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# sort list_contours
	dtype = [('value', int), ('index', int)]
	arr_tmp = np.array([(transfer(cnt), idx) for idx, cnt in enumerate(list_contours)], dtype=dtype)
	arr_tmp = np.sort(arr_tmp, order='value')
	list_contours = list(map(lambda value: list_contours[value[1]], arr_tmp))


	kernel = np.ones((3,3),np.uint8)
	for lc in list_contours:
		x, y, w, h = cv2.boundingRect(lc)
		rois.append(cv2.erode(binary_image[y - margin:y + h + margin, x - margin:x + w + margin], kernel, iterations = 1))

	return np.array([center_to_fill(roi, idx) for idx, roi in enumerate(rois)])

def is_number_region(x,y,w,h,area):
	img_height = binary_image.shape[0]
	center = (x + w/2, y + h/2)
	return x > 0 and x + w < binary_image.shape[1] and h/w >= 1.2 and h/w <= 5.5 and area > 400 and w*h < 4000 and \
			((abs(center[1] - img_height / 4) < img_height / 10) or \
			(abs(center[1] - 3 * img_height / 4) < img_height / 10))


def transfer(cnt):
	x, y, w, h = cv2.boundingRect(cnt)
	img_h = binary_image.shape[0]
	y = y + h/2
	delta_h = abs(img_h/4 - y)
	if (delta_h < img_h/10) :
		return x + pow(img_h/4, 2)
	else:
		return x + pow(3*img_h/4, 2)


def center_to_fill(image, idx):
	f = IMG_HEIGHT/image.shape[0]
	resize_img = cv2.resize(image, None, fx=f, fy=f, interpolation = cv2.INTER_LINEAR)
	bg = np.uint8(np.full((IMG_HEIGHT, IMG_WIDTH), 255))
	margin = ((bg.shape[0] - resize_img.shape[0]) / 2, (bg.shape[1] - resize_img.shape[1]) / 2)
	bg[floor(margin[0]):floor(margin[0]) + resize_img.shape[0], floor(margin[1]):floor(margin[1]) + resize_img.shape[1]] = resize_img
	# write image to file
	name = str(uuid.uuid4())
	cv2.imwrite('{}/segment/{}{}.png'.format(origin_folder, idx, name), bg)
	return bg


def ignore_absent_file(func, path, exc_inf):
    except_instance = exc_inf[1]
    if isinstance(except_instance, FileNotFoundError):
        return
    raise except_instance


shutil.rmtree(origin_folder + '/segment', onerror=ignore_absent_file)
os.makedirs(origin_folder + '/segment')
if __name__ == '__main__':
	segment(cv2.imread(origin_folder + '/Samples/sample ({}).png'.format(13)))


# train: 1, 10, 12, 17, 2, 24, 25, 3, 33, 36, 48, 37, 40, 
# lỗi: 20