import cv2 
import os
import sys
import numpy as np
import argparse
import glob
import math
from gmm import *

def detect_green(img, img_color):

    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    circle = []
    circular = []
    for ctr in contours:
        per = cv2.arcLength(ctr, True)
        area = cv2.contourArea(ctr)
        if per == 0:
            continue
        circularity = 4*math.pi*(area/(per**2))
        
        if 0.2 < circularity < 0.9:
            circle.append(ctr)

    for cir in circle:

        (x, y), radius = cv2.minEnclosingCircle(cir)
        center = (int(x), int(y) - 1)
        radius = int(radius) - 1
        if radius < 10:
            continue
        else:
            radius -= 2
            cv2.circle(img_color, center, radius, (0,255,0), 2)

    # print(img.shape)
    # cv2.imshow("threshold", img_color)
    # cv2.waitKey(0)

def detect_yellow(img, img_color):

    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    circle = []
    circular = []
    for ctr in contours:
        per = cv2.arcLength(ctr, True)
        area = cv2.contourArea(ctr)
        if per == 0:
            continue
        circularity = 4*math.pi*(area/(per**2))
        
        if 0.2 < circularity < 0.9:
            circle.append(ctr)

    raius = []
    center_coord = []
    for cir in circle:

        (x, y), radius = cv2.minEnclosingCircle(cir)
        center = (int(x), int(y) - 1)
        radius = int(radius) - 1
        raius.append(radius)
        center_coord.append(center)

    i = 0
    if len(raius) > 0:
        # print(i, raius)
        i += 1
        radius_ = np.max(raius) + 2
        index = np.argmax(raius)
        if radius_ < 12:
            radius_ = 12
        cv2.circle(img_color, center_coord[index], radius_, (0,255,255), 2)
    # print(img.shape)
    # cv2.imshow("threshold", img_color)
    # cv2.waitKey(0)

def detect_orange(img, img_color):

    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    circle = []
    circular = []
    for ctr in contours:
        per = cv2.arcLength(ctr, True)
        area = cv2.contourArea(ctr)
        if per == 0:
            continue
        circularity = 4*math.pi*(area/(per**2))
        
        if 0.2 < circularity < 0.9:
            circle.append(ctr)

    raius = []
    center_coord = []
    for cir in circle:

        (x, y), radius = cv2.minEnclosingCircle(cir)
        center = (int(x), int(y) - 1)
        radius = int(radius) - 1
        raius.append(radius)
        center_coord.append(center)

    i = 0
    if len(raius) > 0:
        # print(i, raius)
        i += 1
        radius_ = np.max(raius) + 2
        if np.max(raius) > 3:
            # print(radius_-2)
            index = np.argmax(raius)
            if radius_ <= 15:
                radius_ = 12
            cv2.circle(img_color, center_coord[index], radius_, (0,165,255), 2)

    # print(img.shape)
    # cv2.imshow("threshold", img_color)
    # cv2.waitKey(0)


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
        # print(gmm_models)
        # exit(-1) 
    names = ["Green", "Yellow", "Orange"]
    count = 0
    while(cap.isOpened()):
        ret, img = cap.read()
        count += 1
        # if count < 142:
        #         continue
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
            # if count > 142:
            #     mean_val = [ 3e-06, 8e-06, 0.6e-06] #green, yellow, orange threshold
            # else:
            mean_val = [ 3e-06, 8e-06, 0.6e-06] #green, yellow, orange threshold
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

            idx = cv2.cvtColor(idx, cv2.COLOR_BGR2GRAY)
            ret, idx = cv2.threshold(idx,10,255,0)
            # idx = cv2.medianBlur(idx, 5)
            # kernel = np.ones((5,5),np.uint8)
            if k == 1:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            idx = cv2.morphologyEx(idx, cv2.MORPH_OPEN, kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
            idx = cv2.morphologyEx(idx, cv2.MORPH_CLOSE, kernel)
            idx = cv2.dilate(idx,kernel,iterations = 1)

            if k==0:
                detect_green(idx, img)
            elif k==1:
                detect_yellow(idx, img)
            elif k==2:
                detect_orange(idx, img)
                # cv2.imshow(str(k), idx)
                # cv2.waitKey(0)

        # cv2.imshow("threshold", img)
        # cv2.waitKey(0)

            # img_seg = img_seg | idx

        vidWriter.write(img)
    vidWriter.release()
    print("Segmented Video generated. ")    