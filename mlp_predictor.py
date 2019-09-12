"""
After moving all the files using the 1_ file, we run this one to extract
the images from the videos and also create a data file we can use
for training and testing later.
"""
import csv
import glob
import os
import os.path
import numpy as np
from keras.models import load_model

from extractor import Extractor

# get the model.
model = Extractor('Inception_V3_Pool.h5') #'Inception_V3_Pool.h5'

pixel_size = "800x450"


def extract_files(test_folder):
	"""After we have all of our videos split between train and test, and
	all nested within folders representing their classes, we need to
	make a data file that we can reference when training our RNN(s).
	This will let us keep track of image sequences and other parts
	of the training process.
	We'll first need to extract images from each of the videos. We'll
	need to record the following data in the file:
	[train|test], class, filename, nb frames
	Extracting can be done with ffmpeg:
	`ffmpeg -i video.mpg image-%04d.jpg`
	"""
	data_file = []
	folders = [test_folder]

	mlp_model = load_model('mlp-features.features-best.hdf5')
	persons = ['Akib', 'Fahim', 'Mushfiq', 'Shaem']

	for folder in folders:
		folder_full = os.path.join( folder)
		test_files = glob.glob(os.path.join(folder_full, '*.jpg'))

		for test_file in test_files:

			print(test_file)

			filename_without_ext = test_file.split(os.path.sep)[-1].split('.')[0]

			print(filename_without_ext)

			feature = model.extract(test_file)
			feature = np.expand_dims(feature, 0)
			feature = np.expand_dims(feature, 0)

			prediction = mlp_model.predict(feature)
			prediction = prediction[0]
			prediction = prediction/np.sum(prediction)
			print(prediction)
			print(np.amax(prediction))

			#maxPos = prediction.index(max(prediction))
			maxPos = np.where(prediction == np.amax(prediction))
			print(maxPos)
			person = persons[maxPos[0][0]]

			if np.amax(prediction)<0.8:
				person = "unknown"

			print(person)

			data_file.append([filename_without_ext, test_file, person])


	with open('test_result.csv', mode='w', newline='') as fout:
		writer = csv.writer(fout)
		writer.writerows(data_file)


def main():
	"""
	Extract images from videos and build a new file that we
	can use as our data input file. It can have format:
	[train|test], class, filename, nb frames
	"""
	extract_files('validity_check')


if __name__ == '__main__':
	main()
