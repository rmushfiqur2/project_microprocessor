# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
import os
import glob
import csv
from functools import partial
import face_recognition
import pickle

import time

import cv2

import urllib.request as ur
import numpy as np
host_address = "http://192.168.0.101:8080/"
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

staff_list_csv = 'staff_list.txt'
number = -1
endNumber = 30
path_location = ''
minFaceArea = 35000
minAcc = 0.76

add_train_image = 120
add_test_image = 20
main_thread_sleep = False

matching = True
#from extractor import Extractor
# get the model.
#model = Extractor('Inception_V3_Pool.h5') #'Inception_V3_Pool.h5'
#from keras.models import load_model
#mlp_model = load_model('mlp-features.features-best.hdf5')

capture_propic = False
capture_images = False
pause_cam_fun = False

dataset = 'Staffs'
encoded_file = 'encodings_pi.pickle'
train_data = []
names_in_train = []
counts_in_train = []

from encode_faces_pi import encode_faces

import pyttsx3


def loadPickleModel():
	global train_data, names_in_train, counts_in_train
	if os.path.isfile(encoded_file):
		train_data = pickle.loads(open(encoded_file, "rb").read())
		name_list = train_data["names"]
		names_in_train = list(set(name_list))
		counts_in_train = []
		for a_name in names_in_train:
			p = name_list.count(a_name)
			counts_in_train.append(p)

class Thread(QThread):
	#changePixmap = pyqtSignal(QtGui.QImage)
	newShot = pyqtSignal()
	detectionShot = pyqtSignal(int)
	new_hit = pyqtSignal(int)

	def __init__(self, parent=None):
		QThread.__init__(self)
		self.myCamSetup()

	def run(self):
		while True:
			self.myCamFun()

	def myCamSetup(self):
		pass
		# Get the path to the sequence for this video.
		#path = os.path.join(objective + '_' + person)
		# create directory if doesn't exist
		#if not os.path.exists(path):
			#os.mkdir(path)

	def myCamFun(self):
		global number, endNumber, path_location, matching
		global add_train_image, add_test_image
		global minFaceArea
		global user_file
		global graph
		global model, mlp_model
		global capture_propic, capture_images
		global minAcc
		global pause_cam_fun
		try:
			response = ur.urlopen(host_address + "shot.jpg")
		except:
			print("server request time out")
			time.sleep(2)
			return
		image = np.asarray(bytearray(response.read()), dtype="uint8")
		try:
			frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
		except:
			print("finished")
			return
		#frame = cv2.imread('tem.jpg')
		cv2.imwrite('tem.jpg',frame)
		print(frame.shape)
		#time.sleep(0.2)
		height, width, bytesPerComponent = frame.shape
		bytesPerLine = 3 * width
		#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		#cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
		#rawImage = QtGui.QImage(frame.data, width, height, QtGui.QImage.Format_RGB888)
		#rawImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_Indexed8)
		#self.changePixmap.emit(rawImage)

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		faces = face_recognition.face_locations(rgb, model='hog') #(y1, y2, x1, x2)
		#faces = face_cascade.detectMultiScale(gray, 1.3, 5) #((x, y, w, h)
		if len(faces)<1:
			self.newShot.emit()
			return
		for area in faces:
			print("found")
			print(area)
		area = [(y2 - y1) * (x2 - x1) for (y1, x2, y2, x1) in faces]
		max_area = max(area)
		if max_area < minFaceArea:
			print(max_area)
			self.newShot.emit()
			return
		max_index = area.index(max_area)

		face = faces[max_index]

		if capture_images:
			number += 1

		(y1, x2, y2, x1) = face
		# print(box)
		crop_img = frame[y1:y2, x1:x2, :]
		print(crop_img.shape)
		#crop_img = cv2.resize(crop_img, (299, 299))
		#cv2.imshow("cropped", crop_img)
		if capture_images and (number>0) and (number <= endNumber):
			path = os.path.join(path_location, str(number) + '.jpg')
			cv2.imwrite(path, crop_img)
			self.new_hit.emit(2)
		if capture_images and (number > endNumber):
			self.new_hit.emit(10)
		if capture_propic:
			path = os.path.join(path_location, 'propic.jpg')
			cv2.imwrite(path, crop_img)
			self.new_hit.emit(0)
		cv2.imwrite('tem_face.jpg', crop_img)
		#cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		#cv2.imshow("Security Feed", frame)
		#if cv2.waitKey(1) & 0xFF == ord('q'):
			#break'''
		self.newShot.emit()
		if matching:
			print("trying to match")
			encodings = face_recognition.face_encodings(rgb, [face])

			# initialize the list of names for each face detected
			names = []
			# loop over the encodings
			for encoding in encodings:
				# attempt to match each face in the input image to our known
				# encodings
				matches = face_recognition.compare_faces(train_data["encodings"], encoding)
				# print(matches)
				name = "Unknown"

				# check to see if we have found a match
				if True in matches:
					# find the indexes of all matched faces then initialize a
					# dictionary to count the total number of times each face
					# was matched
					matchedIdxs = [i for (i, b) in enumerate(matches) if b]
					counts = {}

					# loop over the matched indexes and maintain a count for
					# each recognized face face
					for i in matchedIdxs:
						name = train_data["names"][i]
						counts[name] = counts.get(name, 0) + 1

					# determine the recognized face with the largest number of
					# votes (note: in the event of an unlikely tie Python will
					# select first entry in the dictionary)
					counts_percent = {}
					for inn, a_name in enumerate(names_in_train):
						score = counts.get(a_name, 0)
						total = counts_in_train[inn]
						counts_percent[a_name] = 100 * score / total

					name = max(counts_percent, key=counts_percent.get)
					acc = counts_percent[name]

					#for key in list(counts_percent.keys()):
						#print("{}: {}".format(key, counts_percent.get(key)))
					#print("[INFO] detected face accuracy: {}".format(acc))

					if acc < 100:
						name = "Unknown"

				print("[INFO] detected face: {}".format(name))

				if name.__eq__("Unknown"):
					self.detectionShot.emit(-1)
				else:
					self.detectionShot.emit(names_in_train.index(name))
				#print(name)
				pause_cam_fun = True
				while pause_cam_fun:
					time.sleep(0.2)
				#print(open_door)


class Ui_MainWindow(object):
	def setupUi(self, MainWindow):
		MainWindow.setObjectName("Last Phase")
		MainWindow.resize(530, 305)
		self.centralwidget = QtWidgets.QWidget(MainWindow)
		self.centralwidget.setObjectName("centralwidget")

		self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
		self.horizontalLayout.setObjectName("horizontalLayout")
		# column 1
		self.verticalLayout = QtWidgets.QVBoxLayout()
		self.verticalLayout.setSpacing(1)
		self.verticalLayout.setObjectName("verticalLayout")
		# column 1, row 1(List)
		self.user_layout=[]
		for user in user_file:
			if len(user)<2:
				continue
			user_name = user[0]
			user_allowed = int(user[1]) > 0.5
			self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
			self.horizontalLayout_2.setSpacing(6)
			self.horizontalLayout_2.setObjectName("horizontalLayout_2"+user_name)
			self.label_3 = QtWidgets.QLabel(self.centralwidget)
			sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
			sizePolicy.setHorizontalStretch(0)
			sizePolicy.setVerticalStretch(0)
			sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
			self.label_3.setSizePolicy(sizePolicy)
			self.label_3.setMaximumSize(QtCore.QSize(45, 45))
			self.label_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
			self.label_3.setText(user_name)
			self.label_3.setPixmap(QtGui.QPixmap(os.path.join('Staffs',user_name,'propic.jpg')))
			self.label_3.setScaledContents(True)
			self.label_3.setObjectName("label_3")
			self.horizontalLayout_2.addWidget(self.label_3)

			self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
			self.checkBox.setObjectName("checkBox")
			self.checkBox.setText(user_name)
			self.checkBox.setChecked(user_allowed)
			self.checkBox.stateChanged.connect(partial(self.user_checked_changed, user_name))
			#self.checkBox.stateChanged['int'].connect(lambda i: self.user_checked_changed(i, user_name))
			self.horizontalLayout_2.addWidget(self.checkBox)

			self.verticalLayout.addLayout(self.horizontalLayout_2)
			self.user_layout.append(self.horizontalLayout_2)
		# column 1, row 2
		self.pushButton = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton.setObjectName("pushButton")
		self.verticalLayout.addWidget(self.pushButton)

		self.horizontalLayout.addLayout(self.verticalLayout)

		# column 2
		self.verticalLayout_2 = QtWidgets.QVBoxLayout()
		self.verticalLayout_2.setObjectName("verticalLayout_2")
		# column 2, row 1
		self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
		self.horizontalLayout_5.setObjectName("horizontalLayout_5")
		# column 2, row 1, 2 columns
		self.label_5 = QtWidgets.QLabel(self.centralwidget)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
		self.label_5.setSizePolicy(sizePolicy)
		self.label_5.setMaximumSize(QtCore.QSize(80, 80))
		self.label_5.setText("")
		self.label_5.setPixmap(QtGui.QPixmap("bf2.jpg"))
		self.label_5.setScaledContents(True)
		self.label_5.setObjectName("label_5")
		self.horizontalLayout_5.addWidget(self.label_5)
		self.label = QtWidgets.QLabel(self.centralwidget)
		self.label.setObjectName("label")
		self.label.setScaledContents(True)
		self.label.setMaximumSize(QtCore.QSize(500, 300))
		#sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		#sizePolicy.setHorizontalStretch(1)
		#sizePolicy.setVerticalStretch(1)
		#sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
		#self.label.setSizePolicy(sizePolicy)
		self.horizontalLayout_5.addWidget(self.label)

		self.verticalLayout_2.addLayout(self.horizontalLayout_5)

		self.label_2 = QtWidgets.QLabel(self.centralwidget)
		self.label_2.setObjectName("label_2")
		font = QtGui.QFont()
		font.setFamily("MS Sans Serif")
		font.setPointSize(10)
		font.setBold(True)
		font.setWeight(75)
		self.label_2.setFont(font)
		self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
		self.label_2.setAlignment(QtCore.Qt.AlignCenter)
		self.verticalLayout_2.addWidget(self.label_2)

		self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
		self.pushButton_2.setObjectName("pushButton_2")
		self.verticalLayout_2.addWidget(self.pushButton_2)

		self.horizontalLayout.addLayout(self.verticalLayout_2)
		self.horizontalLayout.setStretch(0, 1)
		self.horizontalLayout.setStretch(1, 3)

		MainWindow.setCentralWidget(self.centralwidget)
		self.menubar = QtWidgets.QMenuBar(MainWindow)
		self.menubar.setGeometry(QtCore.QRect(0, 0, 530, 21))
		self.menubar.setObjectName("menubar")
		MainWindow.setMenuBar(self.menubar)
		self.statusbar = QtWidgets.QStatusBar(MainWindow)
		self.statusbar.setObjectName("statusbar")
		MainWindow.setStatusBar(self.statusbar)

		self.retranslateUi(MainWindow)
		QtCore.QMetaObject.connectSlotsByName(MainWindow)

		self.pushButton.clicked.connect(lambda: self.printMsg('gre'))

		self.app = app
		self.app.processEvents()
		self.initUI()

	def retranslateUi(self, MainWindow):
		_translate = QtCore.QCoreApplication.translate
		MainWindow.setWindowTitle(_translate("MainWindow", "The last Phase"))
		#self.checkBox.setText(_translate("MainWindow", "Fahim"))
		self.pushButton.setText(_translate("MainWindow", "Train New Person"))
		self.label_5.setText(_translate("MainWindow", "TextLabel"))
		self.label.setText(_translate("MainWindow", "TextLabel"))
		self.label_2.setText(_translate("MainWindow", "Waiting for face\n"))
		self.label.setPixmap(QtGui.QPixmap('bf.jpg'))
		self.label_5.setPixmap(QtGui.QPixmap("bf.jpg"))

	def setupUi2(self, MainWindow):
		centralwidget = QtWidgets.QWidget(MainWindow)
		imglayout = QtWidgets.QHBoxLayout(centralwidget)
		size = 333
		img_8bit = (256 * np.random.random((size, size))).astype(np.uint8)
		img = QtGui.QImage(img_8bit.repeat(4), size, size, QtGui.QImage.Format_RGB32)
		pixmap = QtGui.QPixmap(img)
		self.label = QtWidgets.QLabel()
		self.label.setPixmap(pixmap)
		imglayout.addWidget(self.label)
		MainWindow.setCentralWidget(centralwidget)
		MainWindow.show()
		self.app = app
		self.app.processEvents()
		self.initUI()

	def __init__(self):
		'''super(Ui_MainWindow, self).__init__()
		self.title = 'PyQt4 Video'
		self.left = 100
		self.top = 100
		self.width = 640
		self.height = 480


		imglayout = QtWidgets.QHBoxLayout()
		size = 333
		img_8bit = (256 * np.random.random((size, size))).astype(np.uint8)
		img = QtGui.QImage(img_8bit.repeat(4), size, size, QtGui.QImage.Format_RGB32)
		pixmap = QtGui.QPixmap(img)
		imglabel = QtWidgets.QLabel()
		imglabel.setPixmap(pixmap)
		imglayout.addWidget(imglabel)
		MainWindow.setCentralWidget(imglayout)
		MainWindow.show()
		self.app.processEvents()
		self.initUI()'''

	def initUI(self):
		'''self.setWindowTitle(self.title)
		self.setGeometry(self.left, self.top, self.width, self.height)
		self.resize(800, 600)
		# create a label
		self.label = QLabel(self)
		self.label.move(0, 0)
		self.label.resize(640, 480)'''
		self.th = Thread()
		#self.th.changePixmap.connect(lambda p: self.setPixMap(p))
		self.th.newShot.connect(self.setPixMap0)
		self.th.detectionShot.connect(self.faceDetected)
		self.th.start()

	def setPixMap(self, p):
		#pi = QtGui.QPixmap.fromImage(p)
		#self.label.setPixmap(pi)
		#p = p.scaled(640, 480, Qt.KeepAspectRatio)
		self.label.setPixmap(QtGui.QPixmap('tem.jpg'))
		self.label_5.setPixmap(QtGui.QPixmap('tem_face.jpg'))

	def faceDetected(self, int_code):
		global user_file, names_in_train, pause_cam_fun
		engine = pyttsx3.init()
		if int_code < 0:
			#user = "Unknown"
			engine.say("Authentication Failed.")
			self.label_2.setText("Authentication Failed")
		# time.sleep(5)
		else:
			users = [user[0] for user in user_file if len(user) > 1]
			allowed = [user[1] for user in user_file if len(user) > 1]
			user = names_in_train[int_code]
			index = users.index(user)
			open_door = allowed[index]

			if int(open_door)>0.5:
				engine.say("Hello, "+user + " you can enter now.")
				self.label_2.setText("Hello, "+user + " you can enter now.")
			else:
				engine.say("Sorry, " + user + " you are banned from our company.")
				self.label_2.setText("Sorry, " + user + " you are banned from our company.")
		try:
			engine.endLoop()
		except:
			pass
		engine.runAndWait()
		# run servo
		pause_cam_fun = False


	def setPixMap0(self):
		self.label.setPixmap(QtGui.QPixmap('tem.jpg'))
		self.label_5.setPixmap(QtGui.QPixmap('tem_face.jpg'))

	def showNewStaff(self, user_name):
		self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
		self.horizontalLayout_2.setSpacing(6)
		self.horizontalLayout_2.setObjectName("horizontalLayout_2" + user_name)
		self.label_3 = QtWidgets.QLabel(self.centralwidget)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
		self.label_3.setSizePolicy(sizePolicy)
		self.label_3.setMaximumSize(QtCore.QSize(45, 45))
		self.label_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
		self.label_3.setText(user_name)
		self.label_3.setPixmap(QtGui.QPixmap(os.path.join('Staffs', user_name, 'propic.jpg')))
		self.label_3.setScaledContents(True)
		self.label_3.setObjectName("label_3")
		self.horizontalLayout_2.addWidget(self.label_3)

		self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
		self.checkBox.setObjectName("checkBox")
		self.checkBox.setText(user_name)
		#button.clicked.connect(partial(checkbox.setChecked, user_allowed))
		self.checkBox.stateChanged.connect(partial(self.user_checked_changed, user_name))
		# self.checkBox.stateChanged['int'].connect(lambda i: self.user_checked_changed(i, user_name))
		self.horizontalLayout_2.addWidget(self.checkBox)

		self.verticalLayout.insertLayout(len(self.user_layout),self.horizontalLayout_2)
		self.user_layout.append(self.horizontalLayout_2)


	def printMsg(self, _abc):
		global user_file
		#print("Hello mbangladesh   " + _abc)
		#self.label.setPixmap(QtGui.QPixmap('bf2.jpg'))
		dialog = PopUpDLG(self.th)
		value = dialog.exec_()
		if value is not None:
			users = [user[0] for user in user_file if len(user)>1]
			#allowed = [user[1] for user in user_file if len(user)>1]
			if value not in users:
				user_file.append([value, 0])
				save_user_file()
				self.label_2.setText(value + " has been added successfully.")
				self.showNewStaff(value)


	def user_checked_changed(self, user, int):
		global user_file
		'''if int > 1: # checked
			print(user+ " is allowed")
		else: # unchecked
			print(user + "is banned")'''
		users = [user[0] for user in user_file]
		allowed = [user[1] for user in user_file]
		user_number = users.index(user)
		allowed[user_number] = int//2

		user_file = [[users[i], allowed[i]] for i in range(len(users)) ]
		save_user_file()


class PopUpDLG(QtWidgets.QDialog):
	def __init__(self, th):
		global matching
		self.th = th
		super(PopUpDLG, self).__init__()
		self.setObjectName("self")
		self.resize(200, 71)
		self.setMinimumSize(QtCore.QSize(250, 91))
		self.setMaximumSize(QtCore.QSize(250, 91))
		self.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
		icon = QtGui.QIcon()
		icon.addPixmap(QtGui.QPixmap("bf2.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
		self.setWindowIcon(icon)
		self.gridLayout = QtWidgets.QGridLayout(self)
		self.gridLayout.setObjectName("gridLayout")
		self.text_link = QtWidgets.QLineEdit(self)
		self.text_link.setObjectName("text_link")
		self.gridLayout.addWidget(self.text_link, 0, 0, 1, 2)
		self.progressBar = QtWidgets.QProgressBar(self)
		self.progressBar.setProperty("value", 24)
		self.progressBar.setObjectName("progressBar")
		self.progressBar.setTextVisible(False)
		self.progressBar.setVisible(False)
		self.gridLayout.addWidget(self.progressBar, 1, 0, 1, 2)
		self.add_link = QtWidgets.QPushButton(self)
		self.add_link.setObjectName("add_link")
		self.gridLayout.addWidget(self.add_link, 2, 0, 1, 1)
		self.cancel_link = QtWidgets.QPushButton(self)
		self.cancel_link.setObjectName("cancel_link")
		self.gridLayout.addWidget(self.cancel_link, 2, 1, 1, 1)
		self.retranslateUi(self)
		self.cancel_link.clicked.connect(self.cancel_click)
		self.add_link.clicked.connect(self.get_link)
		self.userName = None
		self.addedUserName = None
		self.propic_needed = True
		self.train_files_done = False
		self.test_files_done = False
		matching = False

		self.th.new_hit.connect(self.update_view)

	def update_view(self, int_code):
		global capture_propic, capture_images
		global number, endNumber
		global add_train_image, add_test_image
		if int_code == 0 :  # propic done
			self.propic_needed = False
			capture_propic = False
			self.cancel_link.setEnabled(True)
			self.add_link.setEnabled(False)
			# self.accept()
		if int_code == 2 : #update progress
			if not self.train_files_done:
				done = add_train_image - (endNumber - number)
				p = int(100*done/add_train_image)
			else:
				done = add_test_image - (endNumber - number)
				p = int(100 * done / add_test_image)
			self.progressBar.setProperty("value", p)
			self.progressBar.setVisible(True)
		if int_code == 10:  # photos done either train or test
			if not self.train_files_done:
				capture_images = False
				self.train_files_done = True
				self.cancel_link.setText("Add Test Data")
				self.cancel_link.setEnabled(True)
			elif not self.test_files_done:
				capture_images = False
				self.test_files_done = True
				self.cancel_link.setText("Start Training")
				self.cancel_link.setEnabled(True)

	def retranslateUi(self, Dialog):
		_translate = QtCore.QCoreApplication.translate
		self.setWindowTitle(_translate("Dialog", "Add Person"))
		self.add_link.setText(_translate("Dialog", "Add"))
		self.cancel_link.setText(_translate("Dialog", "Cancel"))

	def get_link(self):
		global user_file
		global capture_propic
		global path_location
		if self.userName is None:
			self.userName = self.text_link.text()
			if len(self.userName)<2:
				self.userName = None
				return
			self.add_link.setText("Take Profile Photo")
			self.cancel_link.setText("Add Train Data")
			users = [user[0] for user in user_file if len(user)>1]
			if self.userName in users:
				self.propic_needed = False
			if self.propic_needed:
				self.cancel_link.setEnabled(False)
		else:
			path_location = os.path.join("Staffs", self.userName)
			if not os.path.exists(path_location):
				os.mkdir(path_location)
			capture_propic = True
			self.add_link.setEnabled(False)


	def cancel_click(self):
		global capture_images
		global number, endNumber, path_location, add_train_image, add_test_image
		global main_thread_sleep
		global user_file
		if self.userName is None:
			self.reject()
		else:
			if not self.train_files_done:
				num = len(glob.glob(os.path.join("Staffs", self.userName, "train", "*.jpg")))
				number = num
				endNumber = num + add_train_image
				path_location = os.path.join("Staffs", self.userName, "train")
				if not os.path.exists(path_location):
					os.mkdir(path_location)
				capture_images = True
				main_thread_sleep = True
				self.cancel_link.setEnabled(False)
				return

			if not self.test_files_done:
				self.progressBar.setProperty("value", 0)
				self.progressBar.setVisible(False)
				num = len(glob.glob(os.path.join("Staffs", self.userName, "test", "*.jpg")))
				number = num
				endNumber = num + add_test_image
				path_location = os.path.join("Staffs", self.userName, "test")
				if not os.path.exists(path_location):
					os.mkdir(path_location)
				capture_images = True
				main_thread_sleep = True
				self.cancel_link.setEnabled(False)
				return
			else:
				self.progressBar.setProperty("value", 0)
				self.progressBar.setProperty("minimum", 0)
				self.progressBar.setProperty("maximum", 0)
				self.progressBar.setVisible(True)
				self.cancel_link.setEnabled(False)
				# do training now
				# train files
				encode_faces()
				loadPickleModel()
				self.addedUserName = self.userName
				self.accept()


	def exec_(self):
		global matching
		super(PopUpDLG, self).exec_()
		matching = True
		return self.addedUserName


def get_users_list():
	global user_file
	if os.path.exists(staff_list_csv):
		with open(os.path.join(staff_list_csv), 'r') as fin:
			reader = csv.reader(fin, delimiter='\t')
			user_file = list(reader)

	users = [user[0] for user in user_file if len(user)>1]
	allowed = [user[1] for user in user_file if len(user)>1]

	print(users)

	users_path = glob.glob("Staffs/*")

	for a_user in users_path:
		user = a_user.split(os.path.sep)[-1]
		if (user not in users) and (user!='Others'):
			user_file.append([user, 0])
			print(user + " added from database")
	save_user_file()


def save_user_file():
	global user_file
	with open(staff_list_csv, mode='w', newline='') as fout:
		writer = csv.writer(fout, delimiter='\t')
		writer.writerows(user_file)


user_file = []

if __name__ == "__main__":
	loadPickleModel()
	get_users_list()
	import sys
	app = QtWidgets.QApplication(sys.argv)
	MainWindow = QtWidgets.QMainWindow()
	ui = Ui_MainWindow()
	ui.setupUi(MainWindow)
	#ui.setupUi2(MainWindow)
	MainWindow.show()
	sys.exit(app.exec_())
