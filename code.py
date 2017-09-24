import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import torch.utils.data as utils
import glob
import cv2
from os import listdir
from os.path import isfile, join
import csv

# Hyper Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001



# Data
traindatalist = []
traindatapath = '/tmp/anjan/datacopy/train/images'
trainimages = (listdir(traindatapath))
trainimagespath = [traindatapath + '/' + i for i in trainimages]
trainlabel = '/tmp/anjan/datacopy/train/train_labels.csv'
with open(trainlabel, 'r') as f:
    reader = csv.reader(f)
    trainlabellist = list(reader)

for i in trainimagespath:
    im = cv2.imread(i)
    traindatalist.append(im)
traindata = np.asarray(traindatalist)

trainlabels = np.asarray(trainlabellist)


# print("---------------------")
# print(len(traindatalist))
# print(type(traindatalist))
#
# print(len(traindatalist[0]))
# print(type(traindatalist[0]))
#
# print(len(traindatalist[5]))
# print(type(traindatalist[5]))
#
# print("---------------------")



testdatalist =[]
testdatapath = '/tmp/anjan/datacopy/test/images'
testimages = (listdir(testdatapath))
testimagespath = [testdatapath + '/' + i for i in testimages]
testlabel = '/tmp/anjan/datacopy/test/test_labels.csv'
with open(testlabel, 'r') as f:
    reader = csv.reader(f)
    testlabellist = list(reader)
for i in testimagespath:
    im = cv2.imread(i)
    testdatalist.append(im)
testdata = np.asarray(testdatalist)
testlabels = np.asarray(testlabellist)

class RSdata(Dataset):
    def __init__(self, datapath, labelpath):


# print("---------------------")
# print(len(testdatalist))
# print(type(testdatalist))
#
# print(len(testdatalist[0]))
# print(type(testdatalist[0]))
#
# print(len(testdatalist[5]))
# print(type(testdatalist[5]))
#
# print("---------------------")


# print(len(trainlabellist))
# print(len(testlabellist))


# print(len(trainimagespath))

# im = cv2.imread(trainimagespath[0])
# print(type(im))
# print(im.shape)


# Variables
# traindata , trainlabels
# testdata , testlabels

# my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])]
# my_y = [np.array([4.]), np.array([2.])]
# print(my_x)
# tensor_x = torch.stack([torch.Tensor(i) for i in my_x])
# print((tensor_x))
# tensor_y = torch.stack([torch.Tensor(i) for i in my_y])

# my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
# my_dataloader = utils.DataLoader(my_dataset) # create your dataloader
