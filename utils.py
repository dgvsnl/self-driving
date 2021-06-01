import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os
import cv2
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam


def getName(s):
    return s.split("\\")[-1]  
    
def importDataInfo(path):
    cols = ["Center", "Left", "Right", "Steering", "Throttle", "Brake", "Speed"]
    data = pd.read_csv(os.path.join(path, "driving_log.csv"), names = cols)
    data["Center"] = data["Center"].apply(getName)
    #print(data.head())
    print("Total number of Train Images: ", data['Center'].shape[0])    
    return data
    
def balanceData(data, display = True):
    nBins = 31
    samplepedBins = 500
    hist, bins = np.histogram(data["Steering"], nBins)
    center = (bins[:-1] + bins[1:])*0.5
    
    if display:
        plt.bar(center, hist, width=0.06)
        plt.plot([-1,1], [samplepedBins, samplepedBins])
        plt.show()
        
    removeIndexlist = []
    for i in range(nBins):
        binDatalist = []
        for j in range(len(data["Steering"])):
            if data["Steering"][j] >= bins[i] and data["Steering"][j] <= bins[i + 1]:
                binDatalist.append(j)
        
        binDatalist = shuffle(binDatalist)
        binDatalist = binDatalist[samplepedBins:]
        removeIndexlist.extend(binDatalist)
    
    data.drop(data.index[removeIndexlist], inplace = True)
    print("Removed Images: ", len(removeIndexlist))
    print("Remaining Images: ", len(data["Center"]))
    
    hist, _ = np.histogram(data["Steering"], nBins)
    if display:
        plt.bar(center, hist, width=0.06)
        plt.plot([-1,1], [samplepedBins, samplepedBins])
        plt.show()
    
    return data

def loadData(path, data):
    imagePath = []
    Steering = []
    
    for i in range(len(data)):
        
        imagePath.append(os.path.join(path, "IMG", data["Center"].iloc[i]))
        Steering.append(float(data["Steering"].iloc[i]))
        
    imagePath = np.asarray(imagePath)
    steering = np.asarray(Steering)
    
    return imagePath, steering

def augmentImage(imgPath, steering):
    img = mpimg.imread(imgPath)
    
    if np.random.rand() > 0.5:
        pan = iaa.Affine(translate_percent = {'x':(-0.1, 0.1), 'y':(-0.1, 0.1)})
        img = pan.augment_image(img)
    
    if np.random.rand() > 0.5:
        zoom = iaa.Affine(scale = (1,1.2))
        img = zoom.augment_image(img)
    
    if np.random.rand() > 0.5:
        brightness = iaa.Multiply((0.4, 1.2))
        img = brightness.augment_image(img)
    
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    
    return img, steering

def preProcessing(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), sigmaX = 0)
    img = cv2.resize(img, (200,66))
    img = img/255
    
    return img

def batchGen(imagesPath, steeringlist, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []
        for i in range(batchSize):
            index = random.randint(0,len(imagesPath)-1)
            if trainFlag:
                img,steering = augmentImage(imagesPath[index], steeringlist[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringlist[index]
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield(np.asarray(imgBatch),np.asanyarray(steeringBatch))
        
def createModel():
    model = Sequential()
    
    model.add(Convolution2D(24, (5,5), (2,2), input_shape=(66,200,3), activation='elu'))
    model.add(Convolution2D(36, (5,5), (2,2), activation='elu'))
    model.add(Convolution2D(48, (5,5), (2,2), activation='elu'))
    model.add(Convolution2D(64, (3,3), activation='elu'))
    model.add(Convolution2D(64, (3,3), activation='elu'))
    
    model.add(Flatten())
    model.add(Dense(1164, activation='elu'))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    
    model.compile(optimizer = Adam(lr = 0.0001), loss = 'mse')
    
    return model
    
        
    
if __name__ == "__main__":
    imgPath = "simulationData\IMG\center_2021_05_31_11_38_15_235.jpg"
    img = preProcessing(mpimg.imread(imgPath))
    plt.imshow(img)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




