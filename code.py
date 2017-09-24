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
from torch.utils.data import Dataset, DataLoader
# Hyper Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001




class RSdata(Dataset):
    def __init__(self, datapath, labelpath):
        self.datapath = datapath
        self.imagenamelist = listdir(self.datapath)
        self.imagepathlist = [datapath + '/' +  i for i in self.imagenamelist]
        self.labelpath = labelpath
        num = 0
        with open(self.labelpath, 'r') as f:
            num = num+1
            print(num)
            reader = csv.reader(f)
            labellist = list(reader)
        self.label = np.asarray(labellist)
    def __len__(self):
        return len(self.imagenamelist)
    def __getitem__(self, idx):
        im = cv2.imread(self.imagepathlist[idx])
        label = self.label[idx]
        sample = {'image' : im, 'label' : label}
        return sample


testloader = RSdata(datapath = '/tmp/anjan/datacopy/test/images' , labelpath = '/tmp/anjan/datacopy/test/test_labels.csv')
trainloader = RSdata(datapath = '/tmp/anjan/datacopy/train/images' , labelpath =  '/tmp/anjan/datacopy/train/train_labels.csv')

class VanillaCNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(3,32,11,11),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        self.layer2 = nn.Sequential(
        nn.Conv2d(32,64,7,7),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        self.layer3 = nn.Sequential(
        nn.Conv2d(64,64,5,5),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        self.layer4 = nn.Sequential(
        nn.Conv2d(64,64,3,3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        # Reshape or do a .view(-1,9216) after this

        self.layer5 = nn.Sequential(
        nn.Linear(9216, 1024),
        nn.Tanh(),
        nn.Linear(1024 , 256),
        nn.HardTanh(),
        # nn.Linear(256 , 30))
        nn.Linear(256 , 15))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(-1,9216)
        out = self.layer5(out)
        return out
# # Data
# traindatalist = []
# traindatapath = c
# trainimages = (listdir(traindatapath))
# trainimagespath = [traindatapath + '/' + i for i in trainimages]
# trainlabel = '/tmp/anjan/datacopy/train/train_labels.csv'
# with open(trainlabel, 'r') as f:
#     reader = csv.reader(f)
#     trainlabellist = list(reader)
#
# for i in trainimagespath:
#     im = cv2.imread(i)
#     traindatalist.append(im)
# traindata = np.asarray(traindatalist)
#
# trainlabels = np.asarray(trainlabellist)
#
#
# # print("---------------------")
# # print(len(traindatalist))
# # print(type(traindatalist))
# #
# # print(len(traindatalist[0]))
# # print(type(traindatalist[0]))
# #
# # print(len(traindatalist[5]))
# # print(type(traindatalist[5]))
# #
# # print("---------------------")
#
#
#
# testdatalist =[]
# testdatapath = '/tmp/anjan/datacopy/test/images'
# testimages = (listdir(testdatapath))
# testimagespath = [testdatapath + '/' + i for i in testimages]
# testlabel = '/tmp/anjan/datacopy/test/test_labels.csv'
# with open(testlabel, 'r') as f:
#     reader = csv.reader(f)
#     testlabellist = list(reader)
# for i in testimagespath:
#     im = cv2.imread(i)
#     testdatalist.append(im)
# testdata = np.asarray(testdatalist)
# testlabels = np.asarray(testlabellist)
#
#
#
#
# # print("---------------------")
# # print(len(testdatalist))
# # print(type(testdatalist))
# #
# # print(len(testdatalist[0]))
# # print(type(testdatalist[0]))
# #
# # print(len(testdatalist[5]))
# # print(type(testdatalist[5]))
# #
# # print("---------------------")
#
#
# # print(len(trainlabellist))
# # print(len(testlabellist))
#
#
# # print(len(trainimagespath))
#
# # im = cv2.imread(trainimagespath[0])
# # print(type(im))
# # print(im.shape)
#
#
# # Variables
# # traindata , trainlabels
# # testdata , testlabels
#
# # my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])]
# # my_y = [np.array([4.]), np.array([2.])]
# # print(my_x)
# # tensor_x = torch.stack([torch.Tensor(i) for i in my_x])
# # print((tensor_x))
# # tensor_y = torch.stack([torch.Tensor(i) for i in my_y])
#
# # my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
# # my_dataloader = utils.DataLoader(my_dataset) # create your dataloader
--------------")
#
#
# # print(len(trainlabellist))
# # print(len(testlabellist))
#
#
# # print(len(trainimagespath))
#
# # im = cv2.imread(trainimagespath[0])
# # print(type(im))
# # print(im.shape)
#
#
# # Variables
# # traindata , trainlabels
# # testdata , testlabels
#
# # my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])]
# # my_y = [np.array([4.]), np.array([2.])]
# # print(my_x)
# # tensor_x = torch.stack([torch.Tensor(i) for i in my_x])
# # print((tensor_x))
# # tensor_y = torch.stack([torch.Tensor(i) for i in my_y])
#
# # my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
# # my_dataloader = utils.DataLoader(my_dataset) # create your dataloader
--------------")
#
#
# # print(len(trainlabellist))
# # print(len(testlabellist))
#
#
# # print(len(trainimagespath))
#
# # im = cv2.imread(trainimagespath[0])
# # print(type(im))
# # print(im.shape)
#
#
# # Variables
# # traindata , trainlabels
# # testdata , testlabels
#
# # my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])]
# # my_y = [np.array([4.]), np.array([2.])]
# # print(my_x)
# # tensor_x = torch.stack([torch.Tensor(i) for i in my_x])
# # print((tensor_x))
# # tensor_y = torch.stack([torch.Tensor(i) for i in my_y])
#
# # my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
# # my_dataloader = utils.DataLoader(my_dataset) # create your dataloader
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
from torch.utils.data import Dataset, DataLoader
# Hyper Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001




class RSdata(Dataset):
    def __init__(self, datapath, labelpath):
        self.datapath = datapath
        self.imagenamelist = listdir(self.datapath)
        self.imagepathlist = [datapath + '/' +  i for i in self.imagenamelist]
        self.labelpath = labelpath
        num = 0
        with open(self.labelpath, 'r') as f:
            num = num+1
            print(num)
            reader = csv.reader(f)
            labellist = list(reader)
        self.label = np.asarray(labellist)
    def __len__(self):
        return len(self.imagenamelist)
    def __getitem__(self, idx):
        im = cv2.imread(self.imagepathlist[idx])
        label = self.label[idx]
        sample = {'image' : im, 'label' : label}
        return sample


testloader = RSdata(datapath = '/tmp/anjan/datacopy/test/images' , labelpath = '/tmp/anjan/datacopy/test/test_labels.csv')
trainloader = RSdata(datapath = '/tmp/anjan/datacopy/train/images' , labelpath =  '/tmp/anjan/datacopy/train/train_labels.csv')

class VanillaCNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(3,32,11,11),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        self.layer2 = nn.Sequential(
        nn.Conv2d(32,64,7,7),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        self.layer3 = nn.Sequential(
        nn.Conv2d(64,64,5,5),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        self.layer4 = nn.Sequential(
        nn.Conv2d(64,64,3,3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        # Reshape or do a .view(-1,9216) after this

        self.layer5 = nn.Sequential(
        nn.Linear(9216, 1024),
        nn.Tanh(),
        nn.Linear(1024 , 256),
        nn.HardTanh(),
        # nn.Linear(256 , 30))
        nn.Linear(256 , 15))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(-1,9216)
        out = self.layer5(out)
        return out
# # Data
# traindatalist = []
# traindatapath = c
# trainimages = (listdir(traindatapath))
# trainimagespath = [traindatapath + '/' + i for i in trainimages]
# trainlabel = '/tmp/anjan/datacopy/train/train_labels.csv'
# with open(trainlabel, 'r') as f:
#     reader = csv.reader(f)
#     trainlabellist = list(reader)
#
# for i in trainimagespath:
#     im = cv2.imread(i)
#     traindatalist.append(im)
# traindata = np.asarray(traindatalist)
#
# trainlabels = np.asarray(trainlabellist)
#
#
# # print("---------------------")
# # print(len(traindatalist))
# # print(type(traindatalist))
# #
# # print(len(traindatalist[0]))
# # print(type(traindatalist[0]))
# #
# # print(len(traindatalist[5]))
# # print(type(traindatalist[5]))
# #
# # print("---------------------")
#
#
#
# testdatalist =[]
# testdatapath = '/tmp/anjan/datacopy/test/images'
# testimages = (listdir(testdatapath))
# testimagespath = [testdatapath + '/' + i for i in testimages]
# testlabel = '/tmp/anjan/datacopy/test/test_labels.csv'
# with open(testlabel, 'r') as f:
#     reader = csv.reader(f)
#     testlabellist = list(reader)
# for i in testimagespath:
#     im = cv2.imread(i)
#     testdatalist.append(im)
# testdata = np.asarray(testdatalist)
# testlabels = np.asarray(testlabellist)
#
#
#
#
# # print("---------------------")
# # print(len(testdatalist))
# # print(type(testdatalist))
# #
# # print(len(testdatalist[0]))
# # print(type(testdatalist[0]))
# #
# # print(len(testdatalist[5]))
# # print(type(testdatalist[5]))
# #
# # print("---------------------")
#
#
# # print(len(trainlabellist))
# # print(len(testlabellist))
#
#
# # print(len(trainimagespath))
#
# # im = cv2.imread(trainimagespath[0])
# # print(type(im))
# # print(im.shape)
#
#
# # Variables
# # traindata , trainlabels
# # testdata , testlabels
#
# # my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])]
# # my_y = [np.array([4.]), np.array([2.])]
# # print(my_x)
# # tensor_x = torch.stack([torch.Tensor(i) for i in my_x])
# # print((tensor_x))
# # tensor_y = torch.stack([torch.Tensor(i) for i in my_y])
#
# # my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
# # my_dataloader = utils.DataLoader(my_dataset) # create your dataloader
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
from torch.utils.data import Dataset, DataLoader
# Hyper Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001




class RSdata(Dataset):
    def __init__(self, datapath, labelpath):
        self.datapath = datapath
        self.imagenamelist = listdir(self.datapath)
        self.imagepathlist = [datapath + '/' +  i for i in self.imagenamelist]
        self.labelpath = labelpath
        num = 0
        with open(self.labelpath, 'r') as f:
            num = num+1
            print(num)
            reader = csv.reader(f)
            labellist = list(reader)
        self.label = np.asarray(labellist)
    def __len__(self):
        return len(self.imagenamelist)
    def __getitem__(self, idx):
        im = cv2.imread(self.imagepathlist[idx])
        label = self.label[idx]
        sample = {'image' : im, 'label' : label}
        return sample


testloader = RSdata(datapath = '/tmp/anjan/datacopy/test/images' , labelpath = '/tmp/anjan/datacopy/test/test_labels.csv')
trainloader = RSdata(datapath = '/tmp/anjan/datacopy/train/images' , labelpath =  '/tmp/anjan/datacopy/train/train_labels.csv')

class VanillaCNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(3,32,11,11),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        self.layer2 = nn.Sequential(
        nn.Conv2d(32,64,7,7),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        self.layer3 = nn.Sequential(
        nn.Conv2d(64,64,5,5),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        self.layer4 = nn.Sequential(
        nn.Conv2d(64,64,3,3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        # Reshape or do a .view(-1,9216) after this

        self.layer5 = nn.Sequential(
        nn.Linear(9216, 1024),
        nn.Tanh(),
        nn.Linear(1024 , 256),
        nn.HardTanh(),
        # nn.Linear(256 , 30))
        nn.Linear(256 , 15))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(-1,9216)
        out = self.layer5(out)
        return out
# # Data
# traindatalist = []
# traindatapath = c
# trainimages = (listdir(traindatapath))
# trainimagespath = [traindatapath + '/' + i for i in trainimages]
# trainlabel = '/tmp/anjan/datacopy/train/train_labels.csv'
# with open(trainlabel, 'r') as f:
#     reader = csv.reader(f)
#     trainlabellist = list(reader)
#
# for i in trainimagespath:
#     im = cv2.imread(i)
#     traindatalist.append(im)
# traindata = np.asarray(traindatalist)
#
# trainlabels = np.asarray(trainlabellist)
#
#
# # print("---------------------")
# # print(len(traindatalist))
# # print(type(traindatalist))
# #
# # print(len(traindatalist[0]))
# # print(type(traindatalist[0]))
# #
# # print(len(traindatalist[5]))
# # print(type(traindatalist[5]))
# #
# # print("---------------------")
#
#
#
# testdatalist =[]
# testdatapath = '/tmp/anjan/datacopy/test/images'
# testimages = (listdir(testdatapath))
# testimagespath = [testdatapath + '/' + i for i in testimages]
# testlabel = '/tmp/anjan/datacopy/test/test_labels.csv'
# with open(testlabel, 'r') as f:
#     reader = csv.reader(f)
#     testlabellist = list(reader)
# for i in testimagespath:
#     im = cv2.imread(i)
#     testdatalist.append(im)
# testdata = np.asarray(testdatalist)
# testlabels = np.asarray(testlabellist)
#
#
#
#
# # print("---------------------")
# # print(len(testdatalist))
# # print(type(testdatalist))
# #
# # print(len(testdatalist[0]))
# # print(type(testdatalist[0]))
# #
# # print(len(testdatalist[5]))
# # print(type(testdatalist[5]))
# #
# # print("---------------------")
#
#
# # print(len(trainlabellist))
# # print(len(testlabellist))
#
#
# # print(len(trainimagespath))
#
# # im = cv2.imread(trainimagespath[0])
# # print(type(im))
# # print(im.shape)
#
#
# # Variables
# # traindata , trainlabels
# # testdata , testlabels
#
# # my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])]
# # my_y = [np.array([4.]), np.array([2.])]
# # print(my_x)
# # tensor_x = torch.stack([torch.Tensor(i) for i in my_x])
# # print((tensor_x))
# # tensor_y = torch.stack([torch.Tensor(i) for i in my_y])
#
# # my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
# # my_dataloader = utils.DataLoader(my_dataset) # create your dataloader
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
from torch.utils.data import Dataset, DataLoader
# Hyper Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001




class RSdata(Dataset):
    def __init__(self, datapath, labelpath):
        self.datapath = datapath
        self.imagenamelist = listdir(self.datapath)
        self.imagepathlist = [datapath + '/' +  i for i in self.imagenamelist]
        self.labelpath = labelpath
        num = 0
        with open(self.labelpath, 'r') as f:
            num = num+1
            print(num)
            reader = csv.reader(f)
            labellist = list(reader)
        self.label = np.asarray(labellist)
    def __len__(self):
        return len(self.imagenamelist)
    def __getitem__(self, idx):
        im = cv2.imread(self.imagepathlist[idx])
        label = self.label[idx]
        sample = {'image' : im, 'label' : label}
        return sample


testloader = RSdata(datapath = '/tmp/anjan/datacopy/test/images' , labelpath = '/tmp/anjan/datacopy/test/test_labels.csv')
trainloader = RSdata(datapath = '/tmp/anjan/datacopy/train/images' , labelpath =  '/tmp/anjan/datacopy/train/train_labels.csv')

class VanillaCNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(3,32,11,11),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        self.layer2 = nn.Sequential(
        nn.Conv2d(32,64,7,7),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        self.layer3 = nn.Sequential(
        nn.Conv2d(64,64,5,5),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        self.layer4 = nn.Sequential(
        nn.Conv2d(64,64,3,3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        # Reshape or do a .view(-1,9216) after this

        self.layer5 = nn.Sequential(
        nn.Linear(9216, 1024),
        nn.Tanh(),
        nn.Linear(1024 , 256),
        nn.HardTanh(),
        # nn.Linear(256 , 30))
        nn.Linear(256 , 15))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(-1,9216)
        out = self.layer5(out)
        return out
# # Data
# traindatalist = []
# traindatapath = c
# trainimages = (listdir(traindatapath))
# trainimagespath = [traindatapath + '/' + i for i in trainimages]
# trainlabel = '/tmp/anjan/datacopy/train/train_labels.csv'
# with open(trainlabel, 'r') as f:
#     reader = csv.reader(f)
#     trainlabellist = list(reader)
#
# for i in trainimagespath:
#     im = cv2.imread(i)
#     traindatalist.append(im)
# traindata = np.asarray(traindatalist)
#
# trainlabels = np.asarray(trainlabellist)
#
#
# # print("---------------------")
# # print(len(traindatalist))
# # print(type(traindatalist))
# #
# # print(len(traindatalist[0]))
# # print(type(traindatalist[0]))
# #
# # print(len(traindatalist[5]))
# # print(type(traindatalist[5]))
# #
# # print("---------------------")
#
#
#
# testdatalist =[]
# testdatapath = '/tmp/anjan/datacopy/test/images'
# testimages = (listdir(testdatapath))
# testimagespath = [testdatapath + '/' + i for i in testimages]
# testlabel = '/tmp/anjan/datacopy/test/test_labels.csv'
# with open(testlabel, 'r') as f:
#     reader = csv.reader(f)
#     testlabellist = list(reader)
# for i in testimagespath:
#     im = cv2.imread(i)
#     testdatalist.append(im)
# testdata = np.asarray(testdatalist)
# testlabels = np.asarray(testlabellist)
#
#
#
#
# # print("---------------------")
# # print(len(testdatalist))
# # print(type(testdatalist))
# #
# # print(len(testdatalist[0]))
# # print(type(testdatalist[0]))
# #
# # print(len(testdatalist[5]))
# # print(type(testdatalist[5]))
# #
# # print("---------------------")
#
#
# # print(len(trainlabellist))
# # print(len(testlabellist))
#
#
# # print(len(trainimagespath))
#
# # im = cv2.imread(trainimagespath[0])
# # print(type(im))
# # print(im.shape)
#
#
# # Variables
# # traindata , trainlabels
# # testdata , testlabels
#
# # my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])]
# # my_y = [np.array([4.]), np.array([2.])]
# # print(my_x)
# # tensor_x = torch.stack([torch.Tensor(i) for i in my_x])
# # print((tensor_x))
# # tensor_y = torch.stack([torch.Tensor(i) for i in my_y])
#
# # my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
# # my_dataloader = utils.DataLoader(my_dataset) # create your dataloader
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
from torch.utils.data import Dataset, DataLoader
# Hyper Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001




class RSdata(Dataset):
    def __init__(self, datapath, labelpath):
        self.datapath = datapath
        self.imagenamelist = listdir(self.datapath)
        self.imagepathlist = [datapath + '/' +  i for i in self.imagenamelist]
        self.labelpath = labelpath
        num = 0
        with open(self.labelpath, 'r') as f:
            num = num+1
            print(num)
            reader = csv.reader(f)
            labellist = list(reader)
        self.label = np.asarray(labellist)
    def __len__(self):
        return len(self.imagenamelist)
    def __getitem__(self, idx):
        im = cv2.imread(self.imagepathlist[idx])
        label = self.label[idx]
        sample = {'image' : im, 'label' : label}
        return sample


testloader = RSdata(datapath = '/tmp/anjan/datacopy/test/images' , labelpath = '/tmp/anjan/datacopy/test/test_labels.csv')
trainloader = RSdata(datapath = '/tmp/anjan/datacopy/train/images' , labelpath =  '/tmp/anjan/datacopy/train/train_labels.csv')

class VanillaCNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(3,32,11,11),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        self.layer2 = nn.Sequential(
        nn.Conv2d(32,64,7,7),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        self.layer3 = nn.Sequential(
        nn.Conv2d(64,64,5,5),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        self.layer4 = nn.Sequential(
        nn.Conv2d(64,64,3,3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        # Reshape or do a .view(-1,9216) after this

        self.layer5 = nn.Sequential(
        nn.Linear(9216, 1024),
        nn.Tanh(),
        nn.Linear(1024 , 256),
        nn.HardTanh(),
        # nn.Linear(256 , 30))
        nn.Linear(256 , 15))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(-1,9216)
        out = self.layer5(out)
        return out
# # Data
# traindatalist = []
# traindatapath = c
# trainimages = (listdir(traindatapath))
# trainimagespath = [traindatapath + '/' + i for i in trainimages]
# trainlabel = '/tmp/anjan/datacopy/train/train_labels.csv'
# with open(trainlabel, 'r') as f:
#     reader = csv.reader(f)
#     trainlabellist = list(reader)
#
# for i in trainimagespath:
#     im = cv2.imread(i)
#     traindatalist.append(im)
# traindata = np.asarray(traindatalist)
#
# trainlabels = np.asarray(trainlabellist)
#
#
# # print("---------------------")
# # print(len(traindatalist))
# # print(type(traindatalist))
# #
# # print(len(traindatalist[0]))
# # print(type(traindatalist[0]))
# #
# # print(len(traindatalist[5]))
# # print(type(traindatalist[5]))
# #
# # print("---------------------")
#
#
#
# testdatalist =[]
# testdatapath = '/tmp/anjan/datacopy/test/images'
# testimages = (listdir(testdatapath))
# testimagespath = [testdatapath + '/' + i for i in testimages]
# testlabel = '/tmp/anjan/datacopy/test/test_labels.csv'
# with open(testlabel, 'r') as f:
#     reader = csv.reader(f)
#     testlabellist = list(reader)
# for i in testimagespath:
#     im = cv2.imread(i)
#     testdatalist.append(im)
# testdata = np.asarray(testdatalist)
# testlabels = np.asarray(testlabellist)
#
#
#
#
# # print("---------------------")
# # print(len(testdatalist))
# # print(type(testdatalist))
# #
# # print(len(testdatalist[0]))
# # print(type(testdatalist[0]))
# #
# # print(len(testdatalist[5]))
# # print(type(testdatalist[5]))
# #
# # print("---------------------")
#
#
# # print(len(trainlabellist))
# # print(len(testlabellist))
#
#
# # print(len(trainimagespath))
#
# # im = cv2.imread(trainimagespath[0])
# # print(type(im))
# # print(im.shape)
#
#
# # Variables
# # traindata , trainlabels
# # testdata , testlabels
#
# # my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])]
# # my_y = [np.array([4.]), np.array([2.])]
# # print(my_x)
# # tensor_x = torch.stack([torch.Tensor(i) for i in my_x])
# # print((tensor_x))
# # tensor_y = torch.stack([torch.Tensor(i) for i in my_y])
#
# # my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
# # my_dataloader = utils.DataLoader(my_dataset) # create your dataloader
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
from torch.utils.data import Dataset, DataLoader
# Hyper Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001




class RSdata(Dataset):
    def __init__(self, datapath, labelpath):
        self.datapath = datapath
        self.imagenamelist = listdir(self.datapath)
        self.imagepathlist = [datapath + '/' +  i for i in self.imagenamelist]
        self.labelpath = labelpath
        num = 0
        with open(self.labelpath, 'r') as f:
            num = num+1
            print(num)
            reader = csv.reader(f)
            labellist = list(reader)
        self.label = np.asarray(labellist)
    def __len__(self):
        return len(self.imagenamelist)
    def __getitem__(self, idx):
        im = cv2.imread(self.imagepathlist[idx])
        label = self.label[idx]
        sample = {'image' : im, 'label' : label}
        return sample


testloader = RSdata(datapath = '/tmp/anjan/datacopy/test/images' , labelpath = '/tmp/anjan/datacopy/test/test_labels.csv')
trainloader = RSdata(datapath = '/tmp/anjan/datacopy/train/images' , labelpath =  '/tmp/anjan/datacopy/train/train_labels.csv')

class VanillaCNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(3,32,11,11),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        self.layer2 = nn.Sequential(
        nn.Conv2d(32,64,7,7),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        self.layer3 = nn.Sequential(
        nn.Conv2d(64,64,5,5),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        self.layer4 = nn.Sequential(
        nn.Conv2d(64,64,3,3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2,2,2))

        # Reshape or do a .view(-1,9216) after this

        self.layer5 = nn.Sequential(
        nn.Linear(9216, 1024),
        nn.Tanh(),
        nn.Linear(1024 , 256),
        nn.HardTanh(),
        # nn.Linear(256 , 30))
        nn.Linear(256 , 15))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(-1,9216)
        out = self.layer5(out)
        return out
# # Data
# traindatalist = []
# traindatapath = c
# trainimages = (listdir(traindatapath))
# trainimagespath = [traindatapath + '/' + i for i in trainimages]
# trainlabel = '/tmp/anjan/datacopy/train/train_labels.csv'
# with open(trainlabel, 'r') as f:
#     reader = csv.reader(f)
#     trainlabellist = list(reader)
#
# for i in trainimagespath:
#     im = cv2.imread(i)
#     traindatalist.append(im)
# traindata = np.asarray(traindatalist)
#
# trainlabels = np.asarray(trainlabellist)
#
#
# # print("---------------------")
# # print(len(traindatalist))
# # print(type(traindatalist))
# #
# # print(len(traindatalist[0]))
# # print(type(traindatalist[0]))
# #
# # print(len(traindatalist[5]))
# # print(type(traindatalist[5]))
# #
# # print("---------------------")
#
#
#
# testdatalist =[]
# testdatapath = '/tmp/anjan/datacopy/test/images'
# testimages = (listdir(testdatapath))
# testimagespath = [testdatapath + '/' + i for i in testimages]
# testlabel = '/tmp/anjan/datacopy/test/test_labels.csv'
# with open(testlabel, 'r') as f:
#     reader = csv.reader(f)
#     testlabellist = list(reader)
# for i in testimagespath:
#     im = cv2.imread(i)
#     testdatalist.append(im)
# testdata = np.asarray(testdatalist)
# testlabels = np.asarray(testlabellist)
#
#
#
#
# # print("---------------------")
# # print(len(testdatalist))
# # print(type(testdatalist))
# #
# # print(len(testdatalist[0]))
# # print(type(testdatalist[0]))
# #
# # print(len(testdatalist[5]))
# # print(type(testdatalist[5]))
# #
# # print("---------------------")
#
#
# # print(len(trainlabellist))
# # print(len(testlabellist))
#
#
# # print(len(trainimagespath))
#
# # im = cv2.imread(trainimagespath[0])
# # print(type(im))
# # print(im.shape)
#
#
# # Variables
# # traindata , trainlabels
# # testdata , testlabels
#
# # my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])]
# # my_y = [np.array([4.]), np.array([2.])]
# # print(my_x)
# # tensor_x = torch.stack([torch.Tensor(i) for i in my_x])
# # print((tensor_x))
# # tensor_y = torch.stack([torch.Tensor(i) for i in my_y])
#
# # my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
# # my_dataloader = utils.DataLoader(my_dataset) # create your dataloader
