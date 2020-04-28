
# GMM Based Segmentation
## Overview
This repository contains the code, dataset and  results(final_video.mp4) for GMM based Segmentation of colored buoy markers in an underwater scenario. The challenge exists in the fact that there are quite a few similar shades of the same color in the video frame. Training a GMM with the help of the Expectation-Maximization algorithm helps us segment the buoys clearly.

## Dependecies to run the code
```
1. Python 3.5
2. Ubuntu 16.04
3. OpenCV 4.2.0
4. Numpy
5. Scipy
6. Matplotlib
```
## Running the code

To run the code, we require the input video file to be present in the same directory as the python files. We have included the input video file in our repository.  
1. Navigate to the directory where the repository is cloned
2. The user would be asked to specify Training / Validation phase. Training the GMM would require to the training data set to be present in the parent directory. We have provided a initial split of the data. The user could also change the split of training and testing data by changing the seed number found in split.py file. 
```
(optional) python3 split.py
```
3. To run validation / training, run the command and choose the respective option. 
```
python3 detection_pipeline.py
```
4.  A video is generated in the parent directory. 

