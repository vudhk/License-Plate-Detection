import numpy as np
import cv2, os
from DataModel import DataModel
from Classifier import Classifier  
from AlgorithmType import AlgorithmType
from datetime import datetime
import time

origin_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
model_stored_folder = origin_folder + "/ModelTraining"

def train(folder_name):
	data_model = load_data(folder_name)
	data_model = suffle_data(data_model)
	hogs = compute_hog(init_hog(), data_model.characters)
	classifier = Classifier(AlgorithmType.SVM, data_model)
	classifier.train(hogs, data_model.labels)
	classifier.save('{}/{}_{}.dat'.format(model_stored_folder, "SVM", int(time.time())))
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

	return DataModel(np.array(characters), np.array(labels))

def suffle_data(data_model):
	#
	#	TO DO
	#
	return data_model

def init_hog(): 
    winSize = (40, 80)
    blockSize = (20, 40)
    blockStride = (10, 20)
    cellSize = (20, 40)
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

def test():
	pass