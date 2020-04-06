import sys
import numpy as np
import cv2
import os
from scipy.stats import multivariate_normal as mvn
np.set_printoptions(threshold=sys.maxsize)
global model, gmm_models, names

def gmm_predict(img, prob, model_pass = None, train = True):
    global model
    if model_pass is not None:
        model_ = model_pass
    else:
        model_ = model
    for i in range(len(model_)):
        # print(model_[str(i)][2])
        # if model_[str(i)][2] == 0:
        #     print(i)
        #     continue
        prob[:,i] = mvn.pdf(img, model_[str(i)][0], model_[str(i)][1])

    if train == True:
        prob = prob /np.sum(prob, axis=1)[:,np.newaxis]

    model = model_
    return prob

def gmm_update(img, prob):
    global model
    #update mean
    for i in range(len(model)):
        prob_x_pt  = np.matmul(prob.T,img)           # prob multplied with img points 
        # print(np.sum(prob, axis = 0)[i])
        model[str(i)][0] = np.array(prob_x_pt[i,:]) / np.sum(prob, axis = 0)[i]
        
    #update covariance
    for i in range(len(model)): # for each model
        x = img - model[str(i)][0]
        probs = prob[:,i]
        probs_diag = np.diag(probs)
        sigma = np.matmul(x.T, np.matmul(probs_diag, x))
        model[str(i)][1] = sigma/np.sum(prob, axis = 0)[i]                 
        model[str(i)][2] = np.linalg.det(model[str(i)][1])
        # print(model[str(i)][2])
        # input()

def gmm_init(name): # passed
    global model
    np.random.seed(40)
    mean_ = {}
    # mean_["Green"] = np.array([110, 190, 110]) #np.array([70, 120, 70])
    mean_["Green"] = np.array([70, 120, 70])
    # mean_["Yellow"] = np.array([110, 240, 230]) #np.array([70, 190, 180])
    mean_["Yellow"] = np.array([70, 190, 200])
    # mean_["Orange"] = np.array([80, 140, 250]) #np.array([30, 90, 200])
    mean_["Orange"] = np.array([30, 90, 200])
    for i in range(2):
        mean = mean_[name] + np.random.randint(60,size=3)
        mat = np.random.randint(0, high = 60, size=(3,3))
        mat = np.matmul(mat, mat.T)
        mat = (mat + mat.T)/2
        mat_det = np.linalg.det(mat)
        model[str(i)] = [mean, mat, mat_det] 

def load_data(directory, name, split="training"):
    filepath = os.path.join(directory, name + "_Buoys_ak")
    files = open(os.path.join(filepath, split + ".txt"), 'r')
    data = [line.rstrip() for line in files.readlines()]

    return data

def test_model(img, ground_truth):
    global gmm_models, names, model 
    prediction = np.zeros((img.shape[0], 3))
    gt_table = {"Green" : 0, "Yellow" : 1, "Orange" : 2}
    for i, name in enumerate(names): 
        model = gmm_models[name]
        prob = np.zeros(img.shape)
        prob = gmm_predict(img, prob, train = False)
        prob = np.amax(prob, axis = 1)
    
        prediction[:,i] = prob


    indices = np.argmax(prediction, axis = 1)

    pass_indices = np.where(indices == gt_table[ground_truth])
    results = len(pass_indices[0])/len(indices)

    return results

def gmm_train():
    global model, gmm_models, names
    gmm_models = {}
    names = ["Green", "Yellow", "Orange"]

    for name in names:
        model = {}
        gmm_init(name)
        directory = "/home/vishnuu/UMD/ENPM673/Perception_Projects/GMM-based-segmentation"
        train_data = load_data(directory, name)
        # print(len(train_data))
        # exit(-1)
        # test_data = load_data(diretory, name, "test")
        n_runs = 10
        while(n_runs):
            # print (n_runs)
            # prev_mean = [model["0"][0], model["1"][0], model["2"][0]]
            # print(prev_mean)
            for path in train_data: 
                img = cv2.imread((path))
                if name == "Yellow":
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    img = img[int(img.shape[0]*0.35) :, :]
                if name == "Orange":
                    cv2.imshow("yell", img)
                    cv2.waitKey(0)
                img = np.reshape(img, (1, img.shape[0]*img.shape[1], 3))
                img = np.squeeze(img)
                if name == "Orange":
                    indices = np.where(img == np.array([255,255,255]))
                # print(indices)
                else:
                    indices = np.where(img == np.array([0]*3))
                img = np.delete(img, indices[0], axis = 0) 
                prob = np.zeros(img.shape)
                prob = gmm_predict(img, prob)
                gmm_update(img, prob)
            n_runs -= 1

        gmm_models[name] = model
        print(model)
    # exit(-1)
    return gmm_models

if __name__=="__main__":
    global model, gmm_models, names
    gmm_models = {}
    names = ["Green", "Yellow", "Orange"]

    for name in names:
        model = {}
        gmm_init(name)
        directory = "/home/vishnuu/UMD/ENPM673/Perception_Projects/GMM-based-segmentation"
        train_data = load_data(directory, name)
        # test_data = load_data(diretory, name, "test")
        n_runs = 5
        while(n_runs):
            # prev_mean = [model["0"][0], model["1"][0], model["2"][0]]
            # print(prev_mean)
            for path in train_data: 
                img = cv2.imread((path))
                img = np.reshape(img, (1, img.shape[0]*img.shape[1], 3))
                img = np.squeeze(img)
                # indices = np.where(img == np.array([255,255,255]))
                indices = np.where(img == np.array([0, 0, 0]))
                img = np.delete(img, indices[0], axis = 0) 
                prob = np.zeros(img.shape)
                prob = gmm_predict(img, prob)
                gmm_update(img, prob)
            n_runs -= 1

        gmm_models[name] = model

    print('Training results: ')
    for name in names:
        test_data = load_data(directory, name, "training")
        results_ = np.array([])
        for path in test_data:
            img = cv2.imread((path))
            img = np.reshape(img, (1, img.shape[0]*img.shape[1], 3))
            img = np.squeeze(img)
            indices = np.where(img == np.array([255,255,255]))
            #indices = np.where(img == np.array([0, 0, 0]))
            img = np.delete(img, indices[0], axis = 0) 
            results = test_model(img, name)
            results_ = np.append(results_, results)
        print(name + ": ", np.mean(results_))    
        
    print('Test results: ')
    for name in names:
        test_data = load_data(directory, name, "test")
        results_ = np.array([])
        for path in test_data:
            img = cv2.imread((path))
            img = np.reshape(img, (1, img.shape[0]*img.shape[1], 3))
            img = np.squeeze(img)
            # indices = np.where(img == np.array([255,255,255]))
            indices = np.where(img == np.array([0, 0, 0]))
            img = np.delete(img, indices[0], axis = 0) 
            results = test_model(img, name)
            results_ = np.append(results_, results)
        print(name + ": ", np.mean(results_))    
        
                
        
