# usr/bin/python -tt

import numpy as np
import cv2, os, time
from Classifier import Classifier  
from AlgorithmType import AlgorithmType
from os.path import dirname

MODELTRAINING_FOLDER = '/ModelTraining'
MODELFILE_STYPE = '{}_{}.dat'	# {algorithm-name}_{time-stamp}.dat
IMG_WIDTH, IMG_HEIGHT = 40, 80

def train(folder_name, algorithm_type):
	dir_root = dirname(dirname(folder_name)) + MODELTRAINING_FOLDER
	characters, labels = load_data(folder_name)
	characters, labels = suffle_data(characters, labels)
	hogs = compute_hog(init_hog(), characters)
	classifier = Classifier(algorithm_type)
	classifier.train(hogs, labels)
	classifier.save(dir_root + '/' + MODELFILE_STYPE.format(AlgorithmType(algorithm_type).name, int(time.time())))
	return classifier


def load_data(folder_name):
	categories = os.listdir(folder_name)
	characters = []
	labels = []
	for cg in categories:
		chars = os.listdir('{}/{}'.format(folder_name, cg))
		for ch in chars:
			img = cv2.imread('{}/{}/{}'.format(folder_name, cg, ch), 0)
			characters.append(img)
			labels.append(ord(cg))
	return np.array(characters), np.array(labels)

def suffle_data(characters, labels):
	#
	#	TO DO
	#
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
	if IMG_WIDTH % 4 != 0 or IMG_HEIGHT % 4 != 0:
		raise Exception('(width, height) of training image must modulus for 4 equal 0.')
	hogs = []
	for char in characters:
		v = hog.compute(char)
		hogs.append(v)
	return np.squeeze(hogs)

def test():
	pass