from __future__ import print_function
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Cuda cores availablity: ", torch.cuda.is_available())
print(device)
x = torch.rand(5, 3).cuda()
print(x)
#
# Set default to whatever device is available
#


#Read the data from csv file
df = pd.read_csv('datasets/segment_training_v5.csv')
print(df.shape)
print(df.describe())
# Select predictors

target_column = ['Segment']
predictors = list(set(list(df.columns))-set(target_column))

print("label column: ", target_column)
print("prediction columns: ", predictors)

x = df[predictors].values
y = df[target_column].values
#Split data into train and test 
x_train, x_test, y_train, y_test = train_test_split(x, y , train_size = 0.7, random_state =  90)
#
print(x_train.shape); print(x_test.shape)

class ANN(nn.Module):
    def __init__(self, input_dim = 3, output_dim = 1):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32,1)
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
model = ANN(input_dim = 3, output_dim = 1)
model.cuda()
#
print(model)

#
# block
#
x_train = torch.from_numpy(x_train)
x_train = x_train.cuda()
y_train = torch.from_numpy(y_train).view(-1,1)
y_train = y_train.cuda()

x_test = torch.from_numpy(x_test)
x_test = x_test.cuda()
y_test = torch.from_numpy(y_test).view(-1,1)
y_test = y_test.cuda()
#
# Block
#
train = torch.utils.data.TensorDataset(x_train,y_train)
test = torch.utils.data.TensorDataset(x_test,y_test)

train_loader = torch.utils.data.DataLoader(train, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = 64, shuffle = True)

# block
#

loss_fn = nn.BCELoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay= 1e-6, momentum = 0.8)

# lines 1 to 6
#epochs = 2000
epochs = 100
epoch_list = []
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

# lines 7 onwards
model.train() # prepare model for training

for epoch in range(epochs):
    trainloss = 0.0
    valloss = 0.0
    
    correct = 0
    total = 0
    for data,target in train_loader:
        data,target = data.cuda(), target.cuda()
        data = Variable(data).float().to(device)
        target = Variable(target).type(torch.FloatTensor)
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
model.eval() 

with torch.no_grad():
    for data, target in test_loader:
        data = Variable(data).float()
        target = Variable(target).type(torch.FloatTensor)

        output = model(data)
        loss = loss_fn(output, target)
        valloss += loss.item()*data.size(0)
        
        predicted = (torch.round(output.data[0]))
        total += len(target)
        correct += (predicted == target).sum()
    
    valloss = valloss/len(test_loader.dataset)
    accuracy = 100 * correct/ float(total)
    print(accuracy)
