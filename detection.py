import numpy as np
import cv2
import math

def videoToImage(fname,tarname):
	cap = cv2.VideoCapture(fname)
	i=0
	while(cap.isOpened()):
		ret,frame = cap.read()
		if ret == False:
			break
		cv2.imwrite(tarname+'/Img'+str(i)+'.jpg',frame)
		i+=1

def maskCreation(img,point,rad):
	crop=img[point[0][1]-rad:point[0][1]+rad,point[0][0]-rad:point[0][0]+rad]
	h,w = crop.shape[:2]
	mask = np.zeros((h,w),np.uint8)
	cv2.circle(mask, (rad,rad), rad, (255,255,255), -1)
	result = cv2.bitwise_and(crop,crop,mask=mask)
	result[np.where((result==[0,0,0]).all(axis=2))] = [255,255,255];
	#cv2.imshow("Mask",result)
	#cv2.waitKey(0)
	return result

def main():
	'''
	fname = './detectbuoy.avi'
	tarname = './Detection'
	videoToImage(fname,tarname)
	'''
	p = []
	c = 0
	while(c<200):
		fname = './Detection/Img'+str(c)+'.jpg'
		crop = False
		cimg = cv2.imread(fname)
		img = cimg.copy()
		posit = []
		
		x_start, y_start, x_end, y_end = 0, 0, 0, 0
		def mouse_drawing(event, x, y, flags, params):
			#global posit, crop
			
			if event == cv2.EVENT_LBUTTONDOWN:
				posit.append((x,y))	
				crop = True
				r_point = [(posit[0][0], posit[0][1]), (posit[1][0], posit[1][1])]
				if len(r_point)==2:
					p.append(r_point)
					print('reached')
					l = math.floor(math.sqrt((r_point[0][0]- r_point[1][0])**2 + (r_point[0][1]- r_point[1][1])**2))
					RO = maskCreation(img,r_point,l)
					#roi = img[r_point[0][1]-l:r_point[0][1]+l, r_point[0][0]-l:r_point[0][0]+l]
					cv2.imwrite('gblob'+str(c)+'.jpg',RO)
		cv2.imshow("output",img)
		cv2.setMouseCallback("output", mouse_drawing,img)
		cv2.waitKey(0)

		c+=1
		print(c)

	

if __name__ == '__main__':
	main()