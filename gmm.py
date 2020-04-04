import numpy as np
import cv2
global model

def gmm_predict():
    global model
    
    pass

def gmm_update():
    global model
    pass

def gmm_init():
    global model
    np.random.seed(0)
    for i in range(3):
        mean = np.random.randint(255,size=3)
        mat = np.random.randint(-50, high = 50, size=(3,3))
        mat = (mat + mat.T)/2
        model[str(i)] = [mean, mat]
    

if __name__=="__main__":
    global model
    model = {}   
    gmm_init()
    train_data = filter_data()
    while(train_data):
        ele = train_data.pop_item()
        
        
