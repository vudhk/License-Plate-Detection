# usr/bin/python -tt

import numpy as np
import cv2, uuid, os
import glob
import shutil

IMG_WIDTH, IMG_HEIGHT = 32, 40

origin_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
img_processing_output = origin_folder + "/ImgProcessOutput"
if not os.path.exists(img_processing_output):
	os.makedirs(img_processing_output)
else:
	files = glob.glob(img_processing_output + '/*')
	for f in files:
	    os.remove(f)

def get_binary_image():
	return binary_image

def get_board_image():
	cv2.drawContours(fixed_size_image, list_coutours, -1, (0,255,0), 1)
	return fixed_size_image

def get_rois():
	return rois


def segment(image):
	global fixed_size_image
	if image.shape[1] >= image.shape[0] * 2 :
		f = 90 / image.shape[0]
		fixed_size_image = cv2.resize(image, None, fx=f, fy=f, interpolation = cv2.INTER_LINEAR)
	else:
		f = 180 / image.shape[0]
		fixed_size_image = cv2.resize(image, None, fx=f, fy=f, interpolation = cv2.INTER_LINEAR)

	cv2.imwrite(img_processing_output + '/1-fixed_size_image.png', fixed_size_image)
	global binary_image
	gray_image = cv2.cvtColor(fixed_size_image, cv2.COLOR_BGR2GRAY)
	#blur_image = cv2.GaussianBlur(gray_image, (5,5), 0)
	blur_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
	cv2.imwrite(img_processing_output + '/1.5-blur_image.png', blur_image)
	__, binary_image = cv2.threshold(blur_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	#binary_image = cv2.adaptiveThreshold(blur_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
	cv2.imwrite(img_processing_output + '/2-binary_image.png', binary_image)

	edge_image = cv2.Canny(blur_image, 30, 200)
	cv2.imwrite(img_processing_output + '/2.5-edged.png', edge_image)

	#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
	#binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
	#cv2.imwrite(img_processing_output + '/3-binary_image_morphology.png', binary_image)

	__, contours, __ = cv2.findContours(edge_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



	img_clone = fixed_size_image.copy()
	cv2.drawContours(img_clone, contours, -1, (0,255,0), 1)
	cv2.imwrite(img_processing_output + '/4-bounding_rect_full.png', img_clone)

	global bound_rects, list_coutours, rois
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

	img_clone = fixed_size_image.copy()
	cv2.drawContours(img_clone, list_coutours, -1, (0,255,0), 2)
	cv2.imwrite(img_processing_output + '/5-bouding_rect_filter.png', img_clone)

	# sort list_coutours
	epsilon = 45
	dtype = [('value', int), ('index', int)]
	arr_tmp = np.array([(value[0] + pow(10 if value[1] <= epsilon else 100, 2), idx) for idx, value in enumerate(bound_rects)], dtype=dtype)
	arr_tmp = np.sort(arr_tmp, order='value')
	list_coutours = list(map(lambda value: list_coutours[value[1]], arr_tmp))


	for lc in list_coutours:
		x, y, w, h = cv2.boundingRect(lc)
		rois.append(binary_image[y:y + h, x:x + w])
	
	for r in rois:
		name = str(uuid.uuid4())
		cv2.imwrite(origin_folder + "/segment/" + name + ".png", r)
	
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


def ignore_absent_file(func, path, exc_inf):
    except_instance = exc_inf[1]
    if isinstance(except_instance, FileNotFoundError):
        return
    raise except_instance


if __name__ == '__main__':
	shutil.rmtree(origin_folder + '/segment', onerror=ignore_absent_file)
	os.makedirs(origin_folder + '/segment')
	#for i in range(14, 2):
	#	print(i)
	segment(cv2.imread(origin_folder + '/Samples/sample ({}).png'.format(40)))


# train: 1, 10, 12, 17, 2, 24, 25, 3, 33, 36, 48, 37, 40, 
# lỗi: 20