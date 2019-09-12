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

from extractor import Extractor

# get the model.
model = Extractor('Inception_V3_Pool.h5') #'Inception_V3_Pool.h5'

pixel_size = "800x450"
sequence_folder = "train_sequences"


def extract_files():
    data_file = []
    folders = ['train', 'test']

    users = glob.glob("Staffs/*")
    print(users)

    for a_user in users:
        user = a_user.split(os.path.sep)[-1]
        print(user)

        for folder in folders:
            class_folder = os.path.join(a_user, folder)
            class_files = glob.glob(os.path.join(class_folder, '*.jpg'))

            for image_path in class_files:
                print(image_path)
                number = image_path.split(os.path.sep)[-1].split('.')[0]

                data_file.append([folder, user, user + '-' + folder + '-' + number])

                # Get the path to the sequence for this video.
                path = os.path.join(sequence_folder, user + '-' + folder + '-' + number)

                # create directory if doesn't exist
                if not os.path.exists(os.path.join(sequence_folder)):
                    os.mkdir(os.path.join(sequence_folder))

                # Check if we already have it.
                if os.path.isfile(path + '.npy'):
                    continue

                feature = model.extract(image_path)

                # Save the sequence.
                np.save(path, feature)

    with open('train_data_file.csv', mode='w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)


def get_nb_frames_for_video(dest):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    generated_files = glob.glob(os.path.join(dest, '*.jpg'))
    return len(generated_files)


def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:
    [train|test], class, filename, nb frames
    """
    extract_files()


if __name__ == '__main__':
    main()
