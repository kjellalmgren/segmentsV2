from __future__ import print_function
from sklearn import preprocessing
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd

#
#
# determine the supported device
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    #
    # *****************************************
    # *********** hard coded for now **********
    # *****************************************
    #device = torch.device("cpu")
    return device
#
# convert a df to tensor to be used in pytorch
def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)
#
# here we go...
#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
#
np.set_printoptions(suppress=True)
#
print("Cuda cores availablity: ", torch.cuda.is_available())
print("device used: {}".format(device))
x = torch.rand(5, 3)
print(x)
#
# Set default to whatever device is available
#
#Read the data from csv file
print("Reading data datasets/segment_training_v5.csv...")
#columns = ['indicies', 'region', 'office', 'revenue', 'region_normalized', 'office_normalized', 'revenue_normalized']
df = pd.read_csv('datasets/segment_training_v5.csv')
print(df.shape)
print(df.describe())
#
#print("Creating column to be used for normalizing...")
#df.insert(3, "region_normalized", 0.0)
#df.insert(4, "office_normalized", 0.0)
#df.insert(5, "revenue_normalized", 0.0)
#
# copy Revenue to Revenue_normalized, Office to Office_normalized and Region to Region_normalized
#
#print("Adding values to normalized columns...")
#for i in df.index:
#    df.at[i, 'region_normalized'] = df.at[i, 'region']
#    df.at[i, 'office_normalized'] = df.at[i, 'office']
#    df.at[i, 'revenue_normalized'] = df.at[i, 'revenue']
#
# Select predictors
target_column = ['segment']
predictors = list(set(list(df.columns))-set(target_column))

print("label column: ", target_column)
print("prediction columns: ", predictors)

x = df[predictors].values
y = df[target_column].values

# Changed from 6 to 3
x1 = df[df.columns[:3]]
print(x1.head(5))
#
# y is the eight's column, label for each row in the dataset (segment)
#
y1 = df.segment
print("y1: ", y1)
#
#Split data into train and test 
x_train, x_test, y_train, y_test = train_test_split(x1, y1, train_size = 0.7, random_state =  90)
#
#
# Select numerical columns which needs to be normalized
#
#print("Start normalizing columns for region, office and revenue...")
#train_norm = x_train[x_train.columns[3:]]
#test_norm = x_test[x_test.columns[3:]]
#
# Normalize Training Data 
#
#std_scale = preprocessing.StandardScaler().fit(train_norm)
#
# Converting numpy array to dataframe and update x_train
#
#x_train_norm = std_scale.transform(train_norm)
#training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns) 
#x_train.update(training_norm_col)
#
#x_test_norm = std_scale.transform(test_norm)
#testing_norm_col = pd.DataFrame(x_test_norm, index=test_norm.index, columns=test_norm.columns) 
#x_test.update(testing_norm_col)
#
# Show result
#
#print(x_train.tail(15))
#print(x_test.tail(15))
# 
#inverses = std_scale.inverse_transform(training_norm_col)
#j = 1
#for inverse in inverses:
    #print(inverse)
    #print("Values: {0:6f} - {1:3.1f} - {2:3.1f} - {3:10.1f}".format(j, inverse[0], inverse[1], inverse[2]))
#    j = j + 1
#
print("Number of records in traning dataset: ", len(x1))
print("Number of records in test dataset: ", len(y1))
#print("Inverse: (75%) ", len(x_train_norm))
#print("Normalized: (75) ", len(inverses))
print("Preprocessing done...")
#
print(x_train.shape); print(x_test.shape)

class ANN(nn.Module):
    def __init__(self, input_dim = 3, output_dim = 3):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32,3)
        self.dropout = nn.Dropout(0.15)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.output_layer(x)
        
        return nn.Sigmoid()(x)
#
model = ANN(input_dim = 3, output_dim = 3)
model.to(device)
#
print(model)

#
# block
#
# denna del har förändrats, vi behöver se över x_train, y_train
# vi ska prova med att plocka bort region, office samt revenue och bara behålla normalized columns
# torch.Tensor(np.array(df))
# x_train = torch.from_numpy(torch.Tensor(np.array(x_train))).cuda()
x_train = df.select_dtypes(include=float).to_numpy()
print(x_train.shape)

# x_train = x_train.cuda()
#y_train = torch.from_numpy(df_to_tensor(y_train)).view(-1,1)
##y_train = torch.from_numpy(torch.Tensor(np.array(y_train))).view(-1,1)
y_train = df.select_dtypes(include=float).to_numpy()
print(y_train.shape)
#y_train = y_train.view(-1,1)
#y_train = y_train.cuda()
np.set_printoptions(suppress=True)
print("x_train...")
print(x_train[:10])
print("----------")
#
#x_test = torch.from_numpy(df_to_tensor(x_test))
x_test = df.select_dtypes(include=float).to_numpy()
#x_test = x_test.cuda()
#y_test = torch.from_numpy(df_to_tensor(y_test)).view(-1,1)
y_test = df.select_dtypes(include=float).to_numpy()
#y_test = y_test.cuda()
#
# Block
#
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)
train = torch.utils.data.TensorDataset(x_train, y_train)
test = torch.utils.data.TensorDataset(x_test, y_test)

train_loader = torch.utils.data.DataLoader(train, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = 64, shuffle = True)
#
# block
#
loss_fn = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay= 1e-6, momentum = 0.8)

# lines 1 to 6
#epochs = 2000
epochs = 3     # default 100
epoch_list = []
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

# lines 7 onwards
model.train().cuda() # prepare model for training

for epoch in range(epochs):
    trainloss = 0.0
    valloss = 0.0
    correct = 0
    total = 0
    for data,target in train_loader:
        data,target = data.cuda(), target.cuda()
        data = Variable(data).float().to(device)
        target = Variable(target).type(torch.FloatTensor).to(device)
        optimizer.zero_grad()
        output = model(data).to(device)
        predicted = (torch.round(output.data[0])).to(device)
        total += len(target)
        correct += (predicted == target).sum()
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        trainloss += loss.item()*data.size(0)

    trainloss = trainloss/len(train_loader.dataset)
    accuracy = 100 * correct / float(total)
    train_acc_list.append(accuracy)
    train_loss_list.append(trainloss)
    print('Epoch: {} \tTraining Loss: {:.4f}\t Acc: {:.2f}%'.format(
        epoch+1, 
        trainloss,
        accuracy
        ))
    epoch_list.append(epoch + 1)

#
# block
#
correct = 0
total = 0
valloss = 0
model.eval().cuda() 

with torch.no_grad():
    for data, target in test_loader:
        data = Variable(data).float().to(device)
        target = Variable(target).type(torch.FloatTensor).to(device)

        output = model(data).to(device)
        loss = loss_fn(output, target).to(device)
        valloss += loss.item()*data.size(0)
        
        predicted = (torch.round(output.data[0])).to(device)
        total += len(target)
        correct += (predicted == target).sum()
    
    valloss = valloss/len(test_loader.dataset)
    accuracy = 100 * correct/ float(total)
    print("Accuracy: {:.2f}".format(accuracy))
