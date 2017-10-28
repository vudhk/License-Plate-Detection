import numpy as np
import cv2, os
from ImgProcessing import processing
from Classifier import Classifier  
from AlgorithmType import AlgorithmType
from DataModel import DataModel
from Train import init_hog, compute_hog


origin_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
model_training_folder = origin_folder + "/ModelTraining"

def experience(expr_folder, classifier = None):
	#classifier = Classifier(AlgorithmType.SVM)

	model_training = os.listdir(model_training_folder)
	model_training = list(map(lambda x : int(x.split('_')[1].split('.')[0]), model_training))
	model_name = "SVM_" + str(max(model_training)) + ".dat"

	#classifier.load('{}/{}'.format(model_training_folder, model_name))

	imgs = os.listdir(expr_folder)
	for img in imgs:
		chars = processing('{}/{}'.format(expr_folder, img))
		data_model = DataModel(np.array(chars))
		hogs = compute_hog(init_hog(), data_model.characters)
		if len(hogs.shape) == 1 :
			hogs = np.array([hogs])
		resp = classifier.predict(hogs)
		print(chr(resp) + " -- " + img)



