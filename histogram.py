import numpy as np
import cv2
import glob
import argparse
import os
import matplotlib.pyplot as plt

def generate_histogram(greenblob, yellowblob, orangeblob):

	dict_names = ['green', 'yellow', 'orange']

	for name in dict_names:

		title = name + '_blobHist'
		if name == 'green':
			directory = greenblob
		elif name == 'yellow':
			directory = yellowblob
		else:
			directory = orangeblob
		files = glob.glob(os.path.join(directory, '*.txt'), recursive=False)
		blue_hist = []
		green_hist = []
		red_hist = []

		for file_ in files:
			# print(file_)
			fileopen = open(file_, 'r')
			lines = [line.rstrip() for line in fileopen]
			title_ = title + '_' + file_.split('/')[-1].split('.')[0]
			for image in lines:
				img = cv2.imread(image)
				img = np.reshape(img, (1, img.shape[0]*img.shape[1], 3))
				img = np.squeeze(img)
				indices = np.where(img == np.array([255,255,255]))
				img = np.delete(img, indices[0], axis = 0)			
				blue_hist.extend(img[:,0].ravel())
				green_hist.extend(img[:,1].ravel())
				red_hist.extend(img[:,2].ravel())


			fig = plt.figure(figsize=(8,6))
			plt.subplot(3,1,1)
			plt.hist(blue_hist, bins=256, range=(0.0, 256.0), color='blue', label='blue channel')
			plt.legend()

			plt.subplot(3,1,2)
			plt.hist(green_hist, bins=256, range=(0.0, 256.0), color='green', label='green channel')
			plt.legend()

			plt.subplot(3,1,3)
			plt.hist(red_hist, bins=256, range=(0.0, 256.0), color='red', label='red channel')
			plt.legend()
			plt.savefig(title_ + '.png')
			plt.close()


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--greenblob', default='./Green_Buoys', help='Path where the green blobs ae stored')
	parser.add_argument('--yellowblob', default='./Yellow_Buoys', help='Path where the yellow blobs ae stored')
	parser.add_argument('--orangeblob', default='./Orange_Buoys', help='Path where the orange blobs ae stored')
	Flags = parser.parse_args()

	generate_histogram(Flags.greenblob, Flags.yellowblob, Flags.orangeblob)
