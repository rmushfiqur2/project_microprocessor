# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import time

#run
# C:\ProgramData\Anaconda3\envs\face_env\python.exe E:/MP_project/encode_faces.py

# arguments
dataset = 'Staffs'
output = 'encodings_pi.pickle'

# grab the paths to the input images in our dataset
imagePaths = list(paths.list_images(dataset))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []
knownFiles = []


def encode_faces():
	global knownEncodings, knownNames, knownFiles
	if os.path.isfile(output):
		train_data = pickle.loads(open(output, "rb").read())
		knownNames = train_data["names"]
		knownEncodings = train_data["encodings"]
		knownFiles = train_data["filenames"]

	# loop over the image paths
	for (i, imagePath) in enumerate(imagePaths):
		if imagePath in knownFiles:
			continue

		name = imagePath.split(os.path.sep)[-3]

		# load the input image and convert it from BGR (OpenCV ordering)
		# to dlib ordering (RGB)
		print(imagePath) # train\Akib\15.jpg
		image = cv2.imread(imagePath)
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# detect the (x, y)-coordinates of the bounding boxes
		# corresponding to each face in the input image
		#boxes = face_recognition.face_locations(rgb,
												#model='hog')
		h,w,c = rgb.shape
		print(h)
		print(w)
		#boxes = [[0,0,224,224]]
		boxes = [[0, h-1, 0, w-1]]
		'''frame = image
		for bbox in boxes:
			(x, y, w, h) = bbox
			cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
		cv2.imshow("Security Feed", frame)
		cv2.waitKey(0)
		cv2.destroyAllWindows()'''

		# compute the facial embedding for the face
		encodings = face_recognition.face_encodings(rgb, boxes)

		# loop over the encodings
		for encoding in encodings:
			# add each encoding + name to our set of known names and
			# encodings
			knownEncodings.append(encoding)
			knownNames.append(name)

	# how many samples ?
	persons = list(set(knownNames))
	for person in persons:
		print("{} appeared: {} times".format(person, knownNames.count(person)))

	# dump the facial encodings + names to disk
	#print("[INFO] serializing encodings...")
	data = {"encodings": knownEncodings, "names": knownNames, "filenames": knownFiles}
	f = open(output, "wb")
	f.write(pickle.dumps(data))


if __name__ == '__main__':
	main()
