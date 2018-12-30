import torchvision
import numpy
import scipy.io as scio
import pickle
import os
from utils import *

img_root = '/Users/eree/Desktop/Data/jpg'
split_root = '/Users/eree/Desktop/Data/setid.mat'
labels_root = '/Users/eree/Desktop/Data/imagelabels.mat'
train_filesname = '/Users/eree/Desktop/dataset/flowers/train/filenames.pickle'

splitid = scio.loadmat(split_root)
labelsid = scio.loadmat(labels_root)

keyoflabelsid = labelsid.keys()
keyofsplitid = splitid.keys()
print(keyoflabelsid)
print(keyofsplitid)
temp = 0
labels = labelsid.get('labels').flatten().tolist()
for i in labels:
    if i == 0:
        temp = temp + 1

print(temp)
trainid = splitid.get('trnid').flatten().tolist()
validid = splitid.get('valid').flatten().tolist()
testid = splitid.get('tstid').flatten().tolist()


filenames = read_pickle_file(train_filesname)
print('load filenames : %d' % len(filenames))
print(filenames)
train_filename = []

print(trainid)
print(testid)

makedataset(trainid, labels, 'train')
makedataset(validid, labels, 'valid')
makedataset(testid, labels, 'test')

traindataset = read_pickle_file('./test.pickle')
print(traindataset)

print(labels[6765])
print(train_filename)

cwd = os.getcwd()
label_dir = os.path.join(cwd, 'labels.pickle')
save_pickle_file(label_dir, labels)
