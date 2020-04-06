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
        cv2.imshow("s", img)
        cv2.waitKey(0)
        nrows = img.shape[0]
        ncols = img.shape[1]
        img_ = np.reshape(img, (1, img.shape[0]*img.shape[1], 3))
        img_ = np.squeeze(img_)
        prediction = np.zeros((img_.shape[0], 3))
        for i, name in enumerate(names):
            curr_model = gmm_models[name]
            prob = np.zeros(img_.shape)
            prob = gmm_predict(img_, prob, model_pass = curr_model,  train=False)
            prob = np.amax(prob, axis = 1)
            prediction[:,i] = prob
        indices = np.argmax(prediction, axis = 1)
        values = []
        for i in range(indices.shape[0]):
            if indices[i] == 2:
                # print(prediction[i, indices[i]])
                values.append(prediction[i, indices[i]])
        
        indices += 1
        mean_val = 7*np.mean(values)
        for i in range(indices.shape[0]):
            if prediction[i, indices[i]-1] > mean_val:
                continue
            else:
                indices[i] = 0
        # print(np.mean(values), np.max(values))
        # print(indices)
        # indices += 1

        # print(np.mean(prediction[indices]), np.max(prediction[indices]))
        indices = np.where(indices==3, 1, 0)
        indices = np.reshape(indices, (img.shape[0], img.shape[1]))
        img2 = np.dstack([img, indices])
        # idx = np.where(img2[:,:,3] == 1, img[:,:,0:3], 0)
        idx1 = (img2[:,:,3] * img2[:,:,0]).astype(np.uint8)
        idx2 = (img2[:,:,3] * img2[:,:,1]).astype(np.uint8)
        idx3 = (img2[:,:,3] * img2[:,:,2]).astype(np.uint8)
        idx = np.dstack([idx1, idx2, idx3])
        cv2.imshow("modified", idx)
        cv2.waitKey(0)
