import sys
import numpy as np
import cv2
import os
from scipy.stats import multivariate_normal as mvn
np.set_printoptions(threshold=sys.maxsize)
global model, gmm_models, names

def gmm_predict(img, prob, train = True):
    global model
    for i in range(len(model)):
        x_u = img-model[str(i)][0]
        x_uT = x_u.T
        sig_inv = np.linalg.inv(model[str(i)][1])
        coeff = 1/(np.sqrt(2*np.pi*model[str(i)][2]))
        # print(x_u)
        # print(sig_inv) 
        # print(np.diag(np.matmul(x_u, np.matmul(sig_inv, x_uT) ) ) )
        # print(coeff)
        # print(model[str(i)][0])
        # input()
        # if(model[str(i)][0].shape == (1,3)):
            
        # prob[:,i] = np.diag(coeff*np.exp(-0.5*np.matmul(x_u ,np.matmul(sig_inv, x_uT))))
        prob[:,i] = mvn.pdf(img, model[str(i)][0], model[str(i)][1])
        # print(prob[:,i])
        # input()
    # assert(np.sum(prob, axis = 1).all == 1,)
    # assert( 
    if train == True:
        prob = prob /np.sum(prob, axis=1)[:,np.newaxis]
    # print(prob)
    # print(np.sum(prob,axis = 0))    
    return prob

def gmm_update(img, prob):
    global model
    #update mean
    for i in range(len(model)):
        prob_x_pt  = np.matmul(prob.T,img)           # prob multplied with img points 
        # print(np.sum(prob, axis = 0)[i])
        model[str(i)][0] = np.array(prob_x_pt[i,:]) / np.sum(prob, axis = 0)[i]
        # print(np.array(prob_x_pt[i,:]) / np.sum(prob, axis = 0)[i])  
        # input()
        # print(model[str(i)][0])
        # exit(0)
    # print('-------------------------------------------------')
    #update covariance
    for i in range(len(model)): # for each model
        x = img - model[str(i)][0]
        probs = prob[:,i]
        probs_diag = np.diag(probs)
        sigma = np.matmul(x.T, np.matmul(probs_diag, x))
        model[str(i)][1] = sigma/np.sum(prob, axis = 0)[i]          
        # for j in range(len(model)): # for dimension 1 in model i
        #     for k in range(len(model)): # for dimension 2 in model i 
        #         # print(img[:,j].reshape(img.shape[0],1).shape)
        #         # print(np.squeeze(prob[:,i]).shape)
                
        #         mean1 = np.array([[model[str(i)][0][j]]])
        #         mean2 = np.array([[model[str(i)][0][k]]])
        #         # cov1 = model[str(i)][1][j,k]
        #         # print (cov1)
        #         # print(mean1)
        #         # print(mean2.shape)
        #         # exit(0)
        #         # print((img[:,k].reshape(img.shape[0],1) - mean2).shape)
        #         # print((img[:,k].reshape(img.shape[0],1) - mean1).shape)
        #         # print(np.matmul(img[:,k].reshape(img.shape[0],1) - mean1, img[:,k].reshape(img.shape[0],1) - mean2))
        #         # print( (prob[:,i].reshape(prob.shape[0],1).T.shape))
        #         # print((np.multiply(img[:,k].reshape(img.shape[0],1) - mean1, img[:,k].reshape(img.shape[0],1) - mean2)).shape)
        #         # print(np.squeeze(np.matmul( prob[:,i].reshape(prob.shape[0],1).T, np.multiply(img[:,k].reshape(img.shape[0],1) - mean1, img[:,k].reshape(img.shape[0],1) - mean2))))
        #         # print(prob[:,i].reshape(prob.shape[0],1).shape)
        #         # print(np.sum(prob[:,i].reshape(prob.shape[0],1) , axis = 1))
        #         model[str(i)][1][j,k] = np.squeeze(np.matmul( prob[:,i].reshape(prob.shape[0],1).T,
        #                                             np.multiply(img[:,k].reshape(img.shape[0],1) - mean2,
        #                                             img[:,j].reshape(img.shape[0],1) - mean1))) / np.sum(prob, axis = 0)[i]  
                # print(model[str(i)][1][j,k])
                # input()
        # print(model[str(i)][1])        
        model[str(i)][2] = np.linalg.det(model[str(i)][1])
        # print(model[str(i)][2])
        # input()

def gmm_init(name): # passed
    global model
    np.random.seed(40)
    mean_ = {}
    mean_["Green"] = np.array([110, 190, 110])
    mean_["Yellow"] = np.array([110, 240, 230])
    mean_["Orange"] = np.array([80, 140, 250])
    for i in range(3):
        mean = mean_[name] + np.random.randint(50,size=3)
        # print(mean)
        mat = np.random.randint(0, high = 60, size=(3,3))
        mat = np.matmul(mat, mat.T)
        mat = (mat + mat.T)/2
        # print(mat)
        mat_det = np.linalg.det(mat)
        # print(mat_det)
        model[str(i)] = [mean, mat, mat_det]
        # print(model[str(i)])
    # print(model[str(2)][0])
    # model["0"][0] = np.array([110,190,110])

def load_data(directory, name, split="training"):
    filepath = os.path.join(directory, name + "_Buoys")
    files = open(os.path.join(filepath, split + ".txt"), 'r')
    data = [line.rstrip() for line in files.readlines()]

    return data

def test_model(img, ground_truth):
    global gmm_models, names, model 
    prediction = np.zeros((img.shape[0], 3))
    gt_table = {"Green" : 0, "Yellow" : 1, "Orange" : 2}
    for i, name in enumerate(names): 
        model = gmm_models[name]
        # print(model["0"][0], model["1"][0], model["2"][0])
        prob = np.zeros(img.shape)
        prob = gmm_predict(img, prob, train = False)
        # print(prob)
        # input()
        prob = np.amax(prob, axis = 1)
        # print(prob.shape)
        prediction[:,i] = prob
    # print(prediction)  
    # input()   
    indices = np.argmax(prediction, axis = 1)
    # print(indices)
    pass_indices = np.where(indices == gt_table[ground_truth])
    results = len(pass_indices[0])/len(indices)

    return results


if __name__=="__main__":
    global model, gmm_models, names
    # img = cv2.imread("/home/vishnuu/UMD/ENPM673/Perception_Projects/GMM-based-segmentation/Orange_Buoys/oblob135.jpg")
    # cv2.imshow("r",img)
    # cv2.waitKey(0)
    gmm_models = {}
    names = ["Green", "Yellow", "Orange"]

    for name in names:
        model = {}
        gmm_init(name)
        directory = "/home/vishnuu/UMD/ENPM673/Perception_Projects/GMM-based-segmentation"
        train_data = load_data(directory, name)
        # test_data = load_data(diretory, name, "test")
        n_runs = 10
        while(n_runs):
            prev_mean = [model["0"][0], model["1"][0], model["2"][0]]
            # print(prev_mean)
            for path in train_data:
                
                # path = train_data.pop(0)
                # print(path)
                # path = "./Green_Buoys/gblob34.jpg" 
                img = cv2.imread((path))
                # cv2.imshow("t",img )
                # cv2.waitKey(0)
                # print(img.shape)
                img = np.reshape(img, (1, img.shape[0]*img.shape[1], 3))
                img = np.squeeze(img)
                indices = np.where(img == np.array([255,255,255]))
                img = np.delete(img, indices[0], axis = 0) 
                # print(img.shape)
                prob = np.zeros(img.shape)
                prob = gmm_predict(img, prob)

                # print(model["0"][0], model["1"][0], model["2"][0])
                gmm_update(img, prob)
            n_runs -= 1

            # curr_mean = [model["0"][0], model["1"][0], model["2"][0]]
            # print(curr_mean - prev_mean)
            # print(np.linalg.norm(np.array(curr_mean - prev_mean), ord=2))
            # if (np.linalg.norm(curr_mean - prev_mean, ord=2)) < 1e-02:
            #     break
        # print('#####################################')
        # print(model["0"][0], model["1"][0], model["2"][0])
        # exit(-1)
        gmm_models[name] = model


            # print(model["0"])
            # print(model["1"])
            # print(model["2"])
    # print(gmm_models["Green"]["0"])
    # print(gmm_models["Green"]["1"])
    # print(gmm_models["Green"]["2"])
    # print(gmm_models["Yellow"]["0"])
    # print(gmm_models["Yellow"]["1"])
    # print(gmm_models["Yellow"]["2"])
    # print(gmm_models["Orange"]["0"])
    # print(gmm_models["Orange"]["1"])
    # print(gmm_models["Orange"]["2"])
    # input()
    for name in names:
        test_data = load_data(directory, name, "test")
        for path in test_data:
            img = cv2.imread((path))
            # cv2.imshow("t",img )
            # cv2.waitKey(0)
            # print(img.shape)
            img = np.reshape(img, (1, img.shape[0]*img.shape[1], 3))
            img = np.squeeze(img)
            indices = np.where(img == np.array([255,255,255]))
            img = np.delete(img, indices[0], axis = 0) 
            # print(img.shape)
            results = test_model(img, name)
            print (results)
        print ("------------------------------------------")
                
        
