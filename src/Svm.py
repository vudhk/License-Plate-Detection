import numpy as np
import cv2

class Svm(object):
	def __init__(self, data_model = None):
		self.data_model = data_model
		self.svm = cv2.ml.SVM_create()


	def find_c_and_grama(self):
		#
		#	TO DO
		#
		c = 12.5
		gamma = 0.5
		return c, gamma


	def predict(self, samples):
		return self.svm.predict(samples)[1].ravel()



	def train(self, hogs, labels):
		if self.data_model == None:
			print("data_model is None!")
		else:
			c, gamma = self.find_c_and_grama()
			self.svm.setC(c)
			self.svm.setGamma(gamma)
			self.svm.setType(cv2.ml.SVM_C_SVC)
			self.svm.setKernel(cv2.ml.SVM_RBF)
			self.svm.train(hogs, cv2.ml.ROW_SAMPLE, labels)
		  

	def save(self, file_name):
		self.svm.save(file_name)


	def load(self, file_name):
		self.svm = cv2.ml.SVM_load(file_name)


