import numpy as np
import cv2
from Svm import Svm
from Ann import Ann
from RTrees import RTrees
from DTrees import DTrees 
from AlgorithmType import AlgorithmType

class Classifier(object):
	def __init__(self, algorithm_type, data_model = None):
		self.algorithm_type = algorithm_type
		self.data_model = data_model

		if self.algorithm_type == AlgorithmType.SVM:
			self.alg = Svm(data_model)
		elif self.algorithm_type == AlgorithmType.ANN:
			self.alg = Ann(data_model)
		elif self.algorithm_type == algorithm_type.RTREES:
			self.alg = RTrees(data_model)
		elif self.algorithm_type == algorithm_type.DTREES:
			self.alg = DTrees(data_model)

	def train(self, hogs, labels):
		if hasattr(self, 'alg'):
			self.alg.train(hogs, labels)		
		
	def predict(self, samples):
		if hasattr(self, 'alg'):
			return self.alg.predict(samples)	
		return None

	def save(self, file_name):
		if hasattr(self, 'alg'):
			self.alg.save(file_name)	

	def load(self, file_name):
		if hasattr(self, 'alg'):
			self.alg.load(file_name)	