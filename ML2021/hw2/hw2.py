# -*- coding: utf-8 -*-
"""MLSpring2021 - HW2-1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/174J-9OaAjfSDCs37vFHtGN9E13XXDnnG

# **Homework 2-1 Phoneme Classification**

## The DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus (TIMIT)
The TIMIT corpus of reading speech has been designed to provide speech data for the acquisition of acoustic-phonetic knowledge and for the development and evaluation of automatic speech recognition systems.

This homework is a multiclass classification task, 
we are going to train a deep neural network classifier to predict the phonemes for each frame from the speech corpus TIMIT.

link: https://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3

## Download Data
Download data from google drive, then unzip it.

You should have `timit_11/train_11.npy`, `timit_11/train_label_11.npy`, and `timit_11/test_11.npy` after running this block.<br><br>
`timit_11/`
- `train_11.npy`: training data<br>
- `train_label_11.npy`: training label<br>
- `test_11.npy`:  testing data<br><br>

**notes: if the google drive link is dead, you can download the data directly from Kaggle and upload it to the workspace**
"""

!gdown --id '1Fh8lXaD9fh8-93kqVzajVEj6enqDSQzt' --output data.zip
!unzip data.zip
!ls

from google.colab import drive
drive.mount('/content/drive')

"""## Preparing Data
Load the training and testing data from the `.npy` file (NumPy array).
"""

import numpy as np

print('Loading data ...')

data_root='./timit_11/'
train = np.load(data_root + 'train_11.npy')
train_label = np.load(data_root + 'train_label_11.npy')
test = np.load(data_root + 'test_11.npy')

print('Size of training data: {}'.format(train.shape))
print('Size of testing data: {}'.format(test.shape))



"""## Create Dataset"""

import torch
from torch.utils.data import Dataset

class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

"""Split the labeled data into a training set and a validation set, you can modify the variable `VAL_RATIO` to change the ratio of validation data."""

VAL_RATIO = 0.2

percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))

"""Create a data loader from the dataset, feel free to tweak the variable `BATCH_SIZE` here."""

BATCH_SIZE = 128

from torch.utils.data import DataLoader

train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

"""Cleanup the unneeded variables to save memory.<br>

**notes: if you need to use these variables later, then you may remove this block or clean up unneeded variables later<br>the data size is quite huge, so be aware of memory usage in colab**
"""

import gc

del train, train_label, train_x, train_y, val_x, val_y
gc.collect()

"""## Create Model

Define model architecture, you are encouraged to change and experiment with the model architecture.
"""

import torch
import torch.nn as nn

# class Layer(nn.Module):
#     def __init__(self,in,out,dropout):
#         super(Classifier, self).__init__()
#         self.layer = nn.Linear(in, out)
#         self.dropout = nn.Dropout(dropout)
#         self.act_fn = nn.LeakyReLU()
#     def forward(self,x):
#         x = layer(x)
#         x = dropout(x)
#         return self.act_fn(x)
    

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(429, 1024)
        self.dropout1 = nn.Dropout(0.2)
        self.BN1 = nn.BatchNorm1d(1024)

        self.layer2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.2)
        self.BN2 = nn.BatchNorm1d(512)

        self.layer3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.2)
        self.BN3 = nn.BatchNorm1d(256)
        
        self.layer4 = nn.Linear(256, 128)
        self.dropout4 = nn.Dropout(0.2)
        self.BN4 = nn.BatchNorm1d(128)

        self.layer5 = nn.Linear(128, 64)
        self.dropout5 = nn.Dropout(0.2)
        self.BN5 = nn.BatchNorm1d(64)

        self.layer6 = nn.Linear(64, 64)
        self.dropout6 = nn.Dropout(0.2)
        self.BN6 = nn.BatchNorm1d(64)

        self.out = nn.Linear(64, 39) 

        self.act_fn = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.act_fn(x)
        x = self.BN1(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.act_fn(x)
        x = self.BN2(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = self.act_fn(x)
        x = self.BN3(x)
        x = self.dropout3(x)

        x = self.layer4(x)
        x = self.act_fn(x)
        x = self.BN4(x)
        x = self.dropout4(x)

        x = self.layer5(x)
        x = self.act_fn(x)
        x = self.BN5(x)
        x = self.dropout5(x)

        x = self.layer6(x)
        x = self.act_fn(x)
        x = self.BN6(x)
        x = self.dropout6(x)

        x = self.out(x)
        
        return x

"""## Training"""

#check device
def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'

"""Fix random seeds for reproducibility."""

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

"""Feel free to change the training parameters here."""

# fix random seed for reproducibility
same_seeds(0)

# get device 
device = get_device()
print(f'DEVICE: {device}')

# training parameters
num_epoch = 100               # number of training epoch
learning_rate = 0.0001       # learning rate

# the path where checkpoint saved
model_path = './model.ckpt'

# create model, define a loss function, and optimizer
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9,weight_decay=0.001)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

# start training

best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train() # set the model to training mode
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad() 
        outputs = model(inputs) 
        batch_loss = criterion(outputs, labels)
        _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        batch_loss.backward() 
        optimizer.step() 

        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()

    # validation
    if len(val_set) > 0:
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels) 
                _, val_pred = torch.max(outputs, 1) 
            
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
        ))

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')

"""## Testing

Create a testing dataset, and load model from the saved checkpoint.
"""

# create testing dataset
test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# create model and load weights from checkpoint
model = Classifier().to(device)
model.load_state_dict(torch.load(model_path))

"""Make prediction."""

predict = []
model.eval() # set the model to evaluation mode
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability

        for y in test_pred.cpu().numpy():
            predict.append(y)

"""Write prediction to a CSV file.

After finish running this block, download the file `prediction.csv` from the files section on the left-hand side and submit it to Kaggle.
"""

with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))