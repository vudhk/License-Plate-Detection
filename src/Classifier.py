import numpy as np
import cv2
from Svm import Svm
import Ann, DTrees 
from AlgorithmType import AlgorithmType

class Classifier(object):
	def __init__(self, algorithm_type):
		if algorithm_type == AlgorithmType.SVM:
			self.alg = Svm()
		elif algorithm_type == AlgorithmType.ANN:
			self.alg = Ann()
		elif algorithm_type == AlgorithmType.DTREES:
			self.alg = DTrees()

	def train(self, hogs, labels):
		if hasattr(self, 'alg'):
			self.alg.train(hogs, labels)		
		
	def predict(self, samples):
		if hasattr(self, 'alg'):
			return self.alg.predict(samples)	
		raise Exception('Can not predict label for this sample')

	def save(self, file_name):
		if hasattr(self, 'alg'):
			self.alg.save(file_name)	

	def load(self, file_name):
		if hasattr(self, 'alg'):
			self.alg.load(file_name)	