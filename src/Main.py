import numpy as np
import cv2, sys, os, Train, Experience, imutils
from AlgorithmType import AlgorithmType
from tkinter import *
from tkinter.filedialog import askopenfilename, Open
from tkinter.ttk import Frame, Button, Style, Entry
from imutils import contours
from PIL import Image, ImageTk
from ImgProcessing import segment

SAMPLE_FOLDER = "/Samples"
DATATRAINING_FOLDER = "/DataTraining"

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
		
		frame = Frame(self, width=400, relief=SUNKEN, borderwidth=1)
		frame.pack(side=LEFT, fill=Y)
		self.pack(fill=BOTH, expand=True)

		global panel, image, txt
		panel = Label(frame)
		panel.pack(side=TOP, fill=BOTH, expand=True)
		browseBtn = Button(frame, width=50, text="Choose Image", command=chooseFile)
		browseBtn.pack(side=BOTTOM, padx=20, pady=20)

		regBtn = Button(self, text="Recognise", command=recognise)
		regBtn.pack(side=LEFT, padx=20)

		txt = Entry(self)
		txt.pack(fill=X, expand=False, side=LEFT)

		training()


	
def chooseFile():
	global image;

	path = askopenfilename()

	if len(path) > 0:
		image = cv2.imread(path)
		image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image1 = Image.fromarray(image1)
		image1 = ImageTk.PhotoImage(image1)
		
		if not panel is None:
			panel.configure(image=image1)
			panel.image = image1
		else:
			print("null")

'''
def segment():
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
	binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
	_, contours, _= cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	chars = []

	for contour in contours:
		(x, y, w, h) = cv2.boundingRect(contour)
		ratio = w/h;
		s = w * h;
		
		imCrop = binary_image[y:y+h, x:x+w]
		white = cv2.countNonZero(imCrop);
		ratioW = white/s;
		
		if ratioW < 0.75 and ratio < 1 and x > 2 and h > image.shape[1]/4 and h < image.shape[1]/2:
			ic = image[y:y+h, x:x+w]
			chars = chars + [ic]

	i = 0
	for c in chars:
		cv2.imshow(str(i),c)
		i = i + 1
'''

def training():
	global model
	# Traning
	model = Train.train(origin_folder + DATATRAINING_FOLDER, AlgorithmType.SVM)


#def main(is_training):
def recognise():
	origin_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

	# Classifier
	result = Experience.experience(image, model)
	txt.insert(0,'')
	txt.insert(0,''.join(result))
	

	#if is_training:
	#	model = Train.train(origin_folder + DATATRAINING_FOLDER)
	#	Experience.experience(origin_folder + SAMPLE_FOLDER, model)
	#else:
	#	Experience.experience(origin_folder + SAMPLE_FOLDER)

	print("")
	print("Completed!")

	return 0

if __name__ == '__main__':
	panel = None;
	root = Tk()
	root.geometry("800x600")
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

