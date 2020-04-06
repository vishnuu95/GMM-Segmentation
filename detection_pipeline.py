import cv2 
import os
import sys
import numpy as np
import argparse
import glob
from gmm import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default = './detectbuoy.avi', help = "Path of the avi video file")
    Flags = parser.parse_args()
    cap = cv2.VideoCapture(Flags.file)
    vidWriter = cv2.VideoWriter("segmented_colors.mp4",cv2.VideoWriter_fourcc(*'mp4v'), 5, (640, 480))
    user_key = int(input("Gmm Training(1)? Validation(2)? Enter 1 / 2 : " ))
    if user_key == 1:
        gmm_models = gmm_train()
        np.savez("gmm_saved_model", gmm_models)
    else:
        load_file = np.load("gmm_saved_model.npz", allow_pickle=True)
        gmm_models = load_file['arr_0'].item()   
    names = ["Green", "Yellow", "Orange"] 
    while(cap.isOpened()):
        ret, img = cap.read()
        if(ret == False):
            break
        # cv2.imshow("s", img)
        # cv2.waitKey(0)
        nrows = img.shape[0]
        ncols = img.shape[1]
        img_ = np.reshape(img, (1, img.shape[0]*img.shape[1], 3))
        img_ = np.squeeze(img_)
        prediction = np.zeros((img_.shape[0], 3))
        for i, name in enumerate(names):
            curr_model = gmm_models[name]
            prob = np.zeros(img_.shape)
            prob = gmm_predict(img_, prob, model_pass = curr_model,  train=False)
            # prob = np.amax(prob, axis = 1)
            prob = np.sum(prob, axis=1)
            prediction[:,i] = prob
        indices = np.argmax(prediction, axis = 1)
        values = []
        img_seg = np.zeros((img.shape), dtype=np.uint8)
        for k in range(3):
            for i in range(indices.shape[0]):
                if indices[i] == k:
                    # print(prediction[i, indices[i]])
                    values.append(prediction[i, indices[i]])
            indices_ = indices.copy()
            # print(np.max(indices))
            indices_ += 1
            mean_val = [ 3e-06, 8e-06, 2e-05] #green, yellow, orange threshold
            # mean_val = 3e-06 # green threshold
            # mean_val = 2e-05
            # print(10*np.mean(values))
            for i in range(indices_.shape[0]):
                if prediction[i, indices_[i]-1] > mean_val[k]:
                    continue
                else:
                    indices_[i] = 0

            # print(np.mean(prediction[indices]), np.max(prediction[indices]))
            indices_ = np.where(indices_== k+1, 1, 0)
            indices_ = np.reshape(indices_, (img.shape[0], img.shape[1]))
            img2 = np.dstack([img, indices_])
            # idx = np.where(img2[:,:,3] == 1, img[:,:,0:3], 0)
            idx1 = (img2[:,:,3] * img2[:,:,0]).astype(np.uint8)
            idx2 = (img2[:,:,3] * img2[:,:,1]).astype(np.uint8)
            idx3 = (img2[:,:,3] * img2[:,:,2]).astype(np.uint8)
            idx = np.dstack([idx1, idx2, idx3])
            # idx = cv2.medianBlur(idx, 5)
            # kernel = np.ones((5,5),np.uint8)
            if k == 1:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            idx = cv2.morphologyEx(idx, cv2.MORPH_OPEN, kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
            idx = cv2.morphologyEx(idx, cv2.MORPH_CLOSE, kernel)
            idx = cv2.dilate(idx,kernel,iterations = 1)
            # idx = cv2.erode(idx,kernel,iterations = 1)
            # idx = np.bitwise_and(img, idx)
            # idx = idx.astype(np.uint8)
            # cv2.imshow("idx", idx)
            # cv2.waitKey(0)
            img_seg = img_seg | idx
        vidWriter.write(img_seg)
    vidWriter.release()
    print("Segmented Video generated. ")    