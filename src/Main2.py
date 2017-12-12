import numpy as np
from ImgProcessing import *
import cv2, sys, os, imutils
from Train import *

SAMPLE_FOLDER = "/Samples"
DATATRAINING_FOLDER = "/DataTraining"
MODELTRAINING_FOLDER = '/ModelTraining'

origin_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
model_training_folder = origin_folder + MODELTRAINING_FOLDER

def experience(image):
	chars = segment(image)
	hogs = compute_hog(init_hog(), np.array(chars))
	if len(hogs.shape) == 1 :
		hogs = np.array([hogs])
	resp = svm.predict(hogs)[1].ravel()
	return list(map(lambda x: chr(x), resp))

if __name__ == '__main__':
	global svm
	svm = cv2.ml.SVM_load(origin_folder + '/ModelTraining/model.dat')
	result = experience(cv2.imread(origin_folder + '/Samples/sample (18).png'))
	print(result)


