# usr/bin/python -tt

import numpy as np
import cv2, os, time, uuid
from os.path import dirname
from math import floor

MODELTRAINING_FOLDER = '/ModelTraining'
#MODELFILE_STYPE = '{}_{}.dat'	# {algorithm-name}_{time-stamp}.dat
IMG_WIDTH, IMG_HEIGHT = 40, 40

def train(folder_name):
	dir_root = dirname(folder_name) + MODELTRAINING_FOLDER
	characters, labels = load_data(folder_name)
	characters, labels = suffle_data(characters, labels)
	hogs = compute_hog(init_hog(), characters)
	# create svm classifier
	svm = cv2.ml.SVM_create()
	svm.setC(12.5)
	svm.setGamma(0.5)
	svm.setType(cv2.ml.SVM_C_SVC)
	svm.setKernel(cv2.ml.SVM_RBF)
	svm.train(hogs, cv2.ml.ROW_SAMPLE, labels)
	svm.save(dir_root + '/model.dat')

def load_data(folder_name):
	categories = os.listdir(folder_name)
	characters = []
	labels = []
	for cg in categories:
		chars = os.listdir('{}/{}'.format(folder_name, cg))
		for ch in chars:
			img = cv2.imread('{}/{}/{}'.format(folder_name, cg, ch), 0)
			characters.append(center_to_fill(img))
			labels.append(ord(cg))
	return np.array(characters), np.array(labels)

def suffle_data(characters, labels):
	return characters, labels

def init_hog():
	winSize = (IMG_WIDTH, IMG_HEIGHT)
	blockSize = (IMG_WIDTH//2, IMG_HEIGHT//2)
	blockStride = (IMG_WIDTH//4, IMG_HEIGHT//4)
	cellSize = (IMG_WIDTH//2, IMG_HEIGHT//2)
	nbins = 9
	derivAperture = 1
	winSigma = -1.
	histogramNormType = 0
	L2HysThreshold = 0.2
	gammaCorrection = 1
	nlevels = 64
	signedGradient = True
	return cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

def compute_hog(hog, characters):
	hogs = []
	for char in characters:
		v = hog.compute(char)
		hogs.append(v)
	return np.squeeze(hogs)

def center_to_fill(image):
	f = IMG_HEIGHT/image.shape[0]
	resize_img = cv2.resize(image, None, fx=f, fy=f, interpolation = cv2.INTER_LINEAR)
	bg = np.uint8(np.full((IMG_HEIGHT, IMG_WIDTH), 255))
	margin = ((bg.shape[0] - resize_img.shape[0]) / 2, (bg.shape[1] - resize_img.shape[1]) / 2)
	bg[floor(margin[0]):floor(margin[0]) + resize_img.shape[0], floor(margin[1]):floor(margin[1]) + resize_img.shape[1]] = resize_img
	return bg


if __name__ == '__main__':
	repo_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
	train(repo_path + '/DataTraining')
