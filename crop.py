import numpy as np
import cv2
import os

def main():
	files = os.listdir('./Yellow_Buoys')
	sorted_files = sorted(files)
	for j,i in enumerate(sorted_files):
		img = cv2.imread('./Yellow_Buoys/'+str(i))
		img2 = img.copy()
		h,w = img2.shape[:2]
		img2 = img2[int(w/2):,:]
		cv2.imwrite('./Yellow_Buoys_New/yblob_half'+str(j)+'.jpg',img2)



if __name__ == '__main__':
	main()