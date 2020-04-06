import numpy as np
import glob
import argparse
import os
# np.random.seed(0)

def generate_split(greenblob, yellowblob, orangeblob):

	greenFiles = np.sort(glob.glob(os.path.join(greenblob, '*.jpg'), recursive=False))
	yellowFiles = np.sort(glob.glob(os.path.join(yellowblob, '*.jpg'), recursive=False))
	orangeFiles = np.sort(glob.glob(os.path.join(orangeblob, '*.jpg'), recursive=False))

	green_training = open(os.path.join(greenblob, 'training.txt'), 'w+')
	green_test = open(os.path.join(greenblob, 'test.txt'), 'w+')
	yellow_training = open(os.path.join(yellowblob, 'training.txt'), 'w+')
	yellow_test = open(os.path.join(yellowblob, 'test.txt'), 'w+')
	orange_training = open(os.path.join(orangeblob, 'training.txt'), 'w+')
	orange_test = open(os.path.join(orangeblob, 'test.txt'), 'w+')

	# names = [greenFiles, yellowFiles, orangeFiles]
	names = {'green':[greenFiles, green_training, green_test], \
		'yellow':[yellowFiles, yellow_training, yellow_test], 'orange':[orangeFiles, orange_training, orange_test]}
	
	for key in names:

		filelist = names[key][0]
		index = np.arange(0, len(names[key][0]))
		np.random.seed(40)
		np.random.shuffle(index)
		i = 0
		training = names[key][1]
		test = names[key][2]
		for idx in index:

			if i < 0.7*int(len(index)):
				training.write(filelist[idx] + ' \n')
			else:
				test.write(filelist[idx] + ' \n')

			i += 1

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--greenblob', default='./Green_Buoys_backup', help='Path where the green blobs ae stored')
	parser.add_argument('--yellowblob', default='./Yellow_Buoys_backup', help='Path where the yellow blobs ae stored')
	parser.add_argument('--orangeblob', default='./Orange_Buoys_backup', help='Path where the orange blobs ae stored')
	Flags = parser.parse_args()

	generate_split(Flags.greenblob, Flags.yellowblob, Flags.orangeblob)
