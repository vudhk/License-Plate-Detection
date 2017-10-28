import numpy as np
import cv2
import Train 
import Experience 
import sys, os

origin_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sample_data_folder = origin_folder + "/Samples"
train_data_folder = origin_folder + "/DataTraining"


#def main(is_training):
def main():
	print("")
	print("========== Recognition License Plate v1.0.0 ===========")
	print("=============== OpenCV version v{} =================".format(cv2.__version__))
	print("")

	classifier = Train.train(train_data_folder)
	Experience.experience(sample_data_folder, classifier)

	#if is_training:
	#	classifier = Train.train(train_data_folder)
	#	Experience.experience(sample_data_folder, classifier)
	#else:
	#	Experience.experience(sample_data_folder)

	print("")
	print("Completed!")

	return 0

if __name__ == '__main__':
	main()
	#if len(sys.argv) > 1:
	#	main(True if str(sys.argv[1]) == "-t" else False)
	#else:
	#	print("Error argument...")
	#	print("Please type command follow: \r\n### python3 Main.py -t|-e")
	#	print("\t-t: for trainning.")
	#	print("\t-e: for experience.")
	#	print("Thank you.")