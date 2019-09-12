# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

#run
# C:\ProgramData\Anaconda3\envs\face_env\python.exe E:/MP_project/recognize_faces_images.py

# arguments
dataset = 'dlib_test'
encoded_file = 'encodings.pickle'

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(dataset))

# load the known faces and embeddings
print("[INFO] loading encodings...")
train_data = pickle.loads(open(encoded_file, "rb").read())


def main():
	not_detected = 0
	not_properly_detected = 0
	correct = 0
	incorrect = 0

	y_true = []
	y_prediction = []

	name_list = train_data["names"]
	names_in_train = set(name_list)
	counts_in_train = []
	for a_name in names_in_train:
		p = name_list.count(a_name)
		counts_in_train.append(p)

	print(names_in_train)
	print(counts_in_train)

	def is_prediction_correct(actual, prediction):
		expected_prediction = actual
		if actual not in names_in_train:
			expected_prediction = "Unknown"
		return expected_prediction.__eq__(prediction)


	def get_percent_acc(a_name, count):
		ind = names_in_train.index(a_name)
		total = counts_in_train[ind]
		return 100*count/total


	# loop over the image paths
	for (i, imagePath) in enumerate(imagePaths):
		# extract the person name from the image path
		print("[INFO] processing image {}/{}".format(i + 1,
													 len(imagePaths)))
		print(imagePath)
		name = imagePath.split(os.path.sep)[-2]
		name_GT = name
		print(name)
		y_true.append(name)

		# load the input image and convert it from BGR (OpenCV ordering)
		# to dlib ordering (RGB)
		image = cv2.imread(imagePath)
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# detect the (x, y)-coordinates of the bounding boxes
		# corresponding to each face in the input image
		boxes = face_recognition.face_locations(rgb,
												model='hog')
		h, w, c = rgb.shape
		# boxes = [[0,0,224,224]]
		#boxes = [[0, h - 1, 0, w - 1]]

		if len(boxes) > 0:
			# find largest bbox
			area = []
			for box in boxes:
				print("area")
				print(box[2] * box[3])
				area.append(box[2] * box[3])
			max_area = max(area)
			max_index = area.index(max_area)

			temp_boxes = [boxes[max_index]]
			boxes = temp_boxes
		else:
			not_detected = not_detected + 1

		frame = image
		for bbox in boxes:
			(x, y, w, h) = bbox
			#cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
		#cv2.imshow("Security Feed", frame)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

		#boxes = [[0,0,224,224]]

		# compute the facial embedding for the face
		encodings = face_recognition.face_encodings(rgb, boxes)

		# initialize the list of names for each face detected
		names = []
		# loop over the encodings
		for encoding in encodings:
			# attempt to match each face in the input image to our known
			# encodings
			matches = face_recognition.compare_faces(train_data["encodings"], encoding)
			#print(matches)
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
					counts_percent[a_name] = 100*score/total

				name = max(counts_percent, key=counts_percent.get)
				acc = counts_percent[name]

				for key in list(counts_percent.keys()):
					print("{}: {}".format(key, counts_percent.get(key)) )
				print("[INFO] detected face accuracy: {}".format(acc))

				if acc<95:
					name = "Unknown"
					not_properly_detected +=1

			print("[INFO] detected face: {}".format(name))
			y_prediction.append(name)
			if is_prediction_correct(name_GT, name):
				correct = correct + 1
			else:
				incorrect = incorrect + 1

	print("no face detectted: {}".format(not_detected))
	print("not properly face detectted: {}".format(not_properly_detected))
	print("correct: {}".format(correct))
	print("in correct: {}".format(incorrect))


if __name__ == '__main__':
	main()
