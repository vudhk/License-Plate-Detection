import numpy as np
import matplotlib
matplotlib.use('Agg')
import cv2, sys, os, imutils
from tkinter import *
from tkinter.filedialog import askopenfilename, Open
from tkinter.ttk import Frame, Button, Style, Entry
from imutils import contours
from PIL import Image, ImageTk
from ImgProcessing import *
from Svm import Svm

SAMPLE_FOLDER = "/Samples"
DATATRAINING_FOLDER = "/DataTraining"
MODELTRAINING_FOLDER = '/ModelTraining'

origin_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
model_training_folder = origin_folder + MODELTRAINING_FOLDER



class MainFrame(Frame):
	def __init__(self, parent):
		Frame.__init__(self, parent)
		self.parent = parent;
		self.initUI()

	def initUI(self):
		print("")
		print("========== Recognition License Plate v1.0.0 ===========")
		print("=============== OpenCV version v{} =================".format(cv2.__version__))
		print("")

		self.parent.title("License Plate Recognition")
		self.style = Style()
		self.style.theme_use("clam")
		global frame2_2
		frame1 = Frame(self, width=400, relief=SUNKEN, borderwidth=1)
		frame2 = Frame(self, width=600, relief=SUNKEN, borderwidth=1)
		frame2_1 = Frame(frame2, width=600, height=200, relief=SUNKEN, borderwidth=0)
		frame2_1_1 = Frame(frame2_1, width=300, height=200, relief=SUNKEN, borderwidth=1)
		frame2_1_2 = Frame(frame2_1, width=300, height=200, relief=SUNKEN, borderwidth=1)
		frame2_2 = Frame(frame2, width=600, height=200, relief=SUNKEN, borderwidth=1)
		frame2_3 = Frame(frame2, width=600, height=200, relief=SUNKEN, borderwidth=1)
		frame1.pack(side=LEFT, fill=Y)
		frame2.pack(side=LEFT, fill=Y)
		frame2_1.pack(fill=X)
		frame2_1_1.pack(side=LEFT)
		frame2_1_2.pack(side=LEFT)
		frame2_2.pack(fill=X)
		frame2_3.pack(fill=X)
		self.pack(fill=BOTH, expand=True)

		global panel1, panel2, panel3, panel4, image, txt
		panel1 = Label(frame1)
		panel2 = Label(frame2_1_1)
		panel3 = Label(frame2_1_2)
		panel1.pack(side=TOP, fill=BOTH, expand=True)
		panel2.pack(side=TOP, fill=BOTH, expand=True)
		panel3.pack(side=TOP, fill=BOTH, expand=True)
		browseBtn = Button(frame1, width=50, text="Choose Image", command=chooseFile)
		browseBtn.pack(side=BOTTOM)

		regBtn = Button(frame2_3, text="Recognise", command=recognise)
		regBtn.pack(side=LEFT)

		txt = Entry(frame2_3, width=100)
		txt.pack(fill=BOTH, expand=False, side=LEFT)

		global svm
		svm = cv2.ml.SVM_load('/ModelTraining/model.dat')


	
def chooseFile():
	global image;

	path = askopenfilename()

	if len(path) > 0:
		image = cv2.imread(path)
		image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image1 = Image.fromarray(image1)
		image1 = ImageTk.PhotoImage(image1)
		
		if not panel1 is None:
			panel1.configure(image=image1)
			panel1.image = image1
		else:
			print("null")
'''
def training():
	global origin_folder
	# Traning
	origin_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
	train(origin_folder + DATATRAINING_FOLDER)
	#model = Classifier.Classifier(AlgorithmType.SVM)
	#model.load('/ModelTraining/SVM_1509192775.dat')
'''

#def main(is_training):
def recognise():

	# Classifier
	result = experience(image, model)

	image1 = ImageTk.PhotoImage(Image.fromarray(get_binary_image()))
	panel2.configure(image=image1)
	panel2.image = image1

	image1 = ImageTk.PhotoImage(Image.fromarray(get_board_image()))
	panel3.configure(image=image1)
	panel3.image = image1

	for roi in get_rois():
		panel4 = Label(frame2_2, width=40)
		image1 = ImageTk.PhotoImage(Image.fromarray(roi))
		panel4.configure(image=image1)
		panel4.image = image1
		panel4.pack(side=LEFT)


	txt.delete(0, len(txt.get()))
	txt.insert(0,''.join(result))
	

	#if is_training:
	#	model = Train.train(origin_folder + DATATRAINING_FOLDER)
	#	Experience.experience(origin_folder + SAMPLE_FOLDER, model)
	#else:
	#	Experience.experience(origin_folder + SAMPLE_FOLDER)

	print("")
	print("Completed!")

	return 0



def experience(image):
	chars = segment(image)
	hogs = compute_hog(init_hog(), np.array(chars))
	if len(hogs.shape) == 1 :
		hogs = np.array([hogs])
	resp = svm.predict(samples)[1].ravel()
	return list(map(lambda x: chr(x), resp))



if __name__ == '__main__':
	panel1 = None;
	root = Tk()
	root.resizable(0,0) 
	root.geometry("1000x600")
	app = MainFrame(root)
	root.mainloop()
	#main()
	#if len(sys.argv) > 1:
	#	main(True if str(sys.argv[1]) == "-t" else False)
	#else:
	#	print("Error argument...")
	#	print("Please type command follow: \r\n### python3 Main.py -t|-e")
	#	print("\t-t: for trainning.")
	#	print("\t-e: for experience.")
	#	print("Thank you.")

